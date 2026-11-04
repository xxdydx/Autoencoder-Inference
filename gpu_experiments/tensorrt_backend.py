"""Standard-ONNX TensorRT backend; no custom TensorRT plugins or fallback.

Gemm retains a single linear+bias boundary, LayerNormalization accumulates in
FP32, Attention is bidirectional, and reconstruction-error scoring is FP32.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch


def export_onnx(model, path, dtype='bf16', length=720):
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper
    typ={'bf16':T.BFLOAT16,'fp16':T.FLOAT16,'fp32':T.FLOAT}[dtype]
    torch_dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[dtype]
    nodes=[];initial=[];counter=0
    def const(value, label):
        if isinstance(value,torch.Tensor):
            value=value.detach().cpu().to(torch_dtype).contiguous()
            tensor=T();tensor.name=label;tensor.data_type=typ
            tensor.dims.extend(value.shape)
            tensor.raw_data=value.view(torch.uint16).numpy().tobytes() if dtype in ('bf16','fp16') else value.numpy().tobytes()
        else:tensor=numpy_helper.from_array(np.asarray(value,dtype=np.int64),label)
        initial.append(tensor);return label
    def op(kind, inputs, label=None, **kwargs):
        nonlocal counter
        counter+=1;label=label or f'{kind}_{counter}'
        nodes.append(h.make_node(kind,inputs,[label],name=label,**kwargs));return label
    def linear(x, layer, name):
        flat=op('Reshape',[x,const([-1,layer.in_features],name+'_flat_shape')])
        y=op('Gemm',[flat,const(layer.weight,name+'_weight'),const(layer.bias,name+'_bias')],
             name+'_gemm',transB=1)
        return op('Reshape',[y,const([-1,length,layer.out_features],name+'_out_shape')])
    def norm(x, layer, name):
        return op('LayerNormalization',[x,const(layer.weight,name+'_weight'),const(layer.bias,name+'_bias')],
                  name,axis=-1,epsilon=layer.eps,stash_type=T.FLOAT)
    def block(x, layer, name):
        z=norm(x,layer.norm1,name+'_ln1')
        qkv=linear(z,layer.qkv,name+'_qkv')
        qkv=op('Reshape',[qkv,const([-1,length,3,4,32],name+'_head_shape')])
        qkv=op('Transpose',[qkv],perm=[2,0,3,1,4])
        split=[name+'_q',name+'_k',name+'_v']
        nodes.append(h.make_node('Split',[qkv,const([1,1,1],name+'_split')],split,axis=0,name=name+'_split_qkv'))
        q,k,v=[op('Squeeze',[s,const([0],s+'_axis')]) for s in split]
        if dtype=='fp32':
            # This build has no FP32 tactic for the explicit Attention node.
            # Standard primitives retain a genuine FP32 TensorRT control.
            scaled=op('Mul',[q,const(torch.tensor(32**-.5),name+'_scale')])
            kt=op('Transpose',[k],perm=[0,1,3,2])
            probabilities=op('Softmax',[op('MatMul',[scaled,kt])],axis=-1)
            attention=op('MatMul',[probabilities,v],name+'_attention')
        else:
            # TensorRT rejects ONNX's explicit softmax_precision attribute. Use
            # its supported native precision policy and validate scores.
            attention=op('Attention',[q,k,v],name+'_attention',is_causal=0,scale=32**-.5)
        attention=op('Transpose',[attention],perm=[0,2,1,3])
        attention=op('Reshape',[attention,const([-1,length,128],name+'_merge_shape')])
        x=op('Add',[x,linear(attention,layer.out,name+'_out')])
        ff=linear(norm(x,layer.norm2,name+'_ln2'),layer.ff1,name+'_ff1')
        ff=op('Gelu',[ff],name+'_gelu',approximate='none')
        return op('Add',[x,linear(ff,layer.ff2,name+'_ff2')])
    x=op('Cast',['input'],to=typ)
    position=const(model.pe[:length].unsqueeze(0),'position')
    z=op('Add',[linear(x,model.input,'input_projection'),position])
    for i,layer in enumerate(model.encoder):z=block(z,layer,f'encoder_{i}')
    z=linear(z,model.bottleneck,'bottleneck')
    z=op('Add',[linear(z,model.decoder_input,'decoder_input'),position])
    for i,layer in enumerate(model.decoder):z=block(z,layer,f'decoder_{i}')
    y=op('Cast',[linear(z,model.output,'output_projection')],to=T.FLOAT)
    diff=op('Sub',[y,'input']);squared=op('Mul',[diff,diff])
    op('ReduceMean',[squared,const([1,2],'score_axes')],'scores',keepdims=0)
    graph=h.make_graph(nodes,'telemetry_autoencoder',[h.make_tensor_value_info('input',T.FLOAT,['batch',length,32])],
                       [h.make_tensor_value_info('scores',T.FLOAT,['batch'])],initial)
    proto=h.make_model(graph,opset_imports=[h.make_opsetid('',23)],ir_version=10,
                       producer_name='cpu-inf-autoencoder-standard-onnx')
    onnx.checker.check_model(proto,full_check=True)
    onnx.save(proto,str(path))
    return {'nodes':len(nodes),'attention_nodes':0 if dtype=='fp32' else 5,
            'attention_form':'primitive' if dtype=='fp32' else 'Attention',
            'opset':23,'dtype':dtype,'length':length}


def build_engine(model, directory, dtype, batch, length=720):
    import tensorrt as trt
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    tag=f'{dtype}_b{batch}_s{length}'
    onnx_path=directory/(tag+'.onnx');plan_path=directory/(tag+'.plan')
    metadata_path=directory/(tag+'_build.json')
    if plan_path.exists():
        metadata=json.loads(metadata_path.read_text());metadata['reused_plan']=True
        return plan_path,metadata
    start=time.perf_counter();export=export_onnx(model,onnx_path,dtype,length)
    export_s=time.perf_counter()-start
    logger=trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger,'')
    builder=trt.Builder(logger);builder.max_threads=2
    network=builder.create_network(0)
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
    config.builder_optimization_level=5
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    profile=builder.create_optimization_profile()
    profile.set_shape('input',(1,length,32),(batch,length,32),(batch,length,32))
    config.add_optimization_profile(profile)
    cache_path=directory/(dtype+'_timing.cache')
    cache_exists=cache_path.exists()
    timing=config.create_timing_cache(cache_path.read_bytes() if cache_exists else b'')
    config.set_timing_cache(timing,False)
    start=time.perf_counter();serialized=builder.build_serialized_network(network,config)
    build_s=time.perf_counter()-start
    if serialized is None:raise RuntimeError('TensorRT returned no serialized engine')
    plan_path.write_bytes(bytes(serialized))
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    metadata=dict(tensorrt=trt.__version__,export=export,export_s=export_s,build_s=build_s,
                  plan_bytes=plan_path.stat().st_size,workspace_limit_bytes=1<<30,optimization_level=5,
                  tf32=False,timing_cache_reused=cache_exists,reused_plan=False,
                  shapes={'min':[1,length,32],'opt':[batch,length,32],'max':[batch,length,32]})
    metadata_path.write_text(json.dumps(metadata,indent=2))
    return plan_path,metadata


class EngineScore:
    def __init__(self,path,graph=True):
        import tensorrt as trt
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('Cannot deserialize TensorRT engine')
        self.graph=graph;self.entries={}
        assert self.engine.get_tensor_dtype('input')==trt.float32
        assert self.engine.get_tensor_dtype('scores')==trt.float32
    def entry(self,n,length):
        key=(n,length)
        if key not in self.entries:
            context=self.engine.create_execution_context()
            assert context.set_input_shape('input',(n,length,32))
            x=torch.zeros(n,length,32,device='cuda',dtype=torch.float32)
            y=torch.empty(n,device='cuda',dtype=torch.float32)
            context.set_tensor_address('input',x.data_ptr())
            context.set_tensor_address('scores',y.data_ptr())
            stream=torch.cuda.current_stream().cuda_stream
            for _ in range(3):assert context.execute_async_v3(stream_handle=stream)
            torch.cuda.synchronize()
            graph=None
            if self.graph:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    assert context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
            self.entries[key]=(context,x,y,graph)
        return self.entries[key]
    def __call__(self,data):
        context,x,y,graph=self.entry(len(data),data.shape[1])
        if graph is not None:
            x.copy_(data);graph.replay()
        else:
            context.set_tensor_address('input',data.data_ptr())
            assert context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        return y
    def inspect(self,path,batch,length=720):
        import tensorrt as trt
        context,_,_,_=self.entry(batch,length)
        inspector=self.engine.create_engine_inspector();inspector.execution_context=context
        Path(path).write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
