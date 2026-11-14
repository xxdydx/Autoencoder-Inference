"""Tune TensorRT and compare it with custom paths using one matched protocol."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from gpu.optimization_study import save, stats


def build(model, directory, dtype, batch, shape_mode):
    import tensorrt as trt
    from gpu.tensorrt_backend import export_onnx
    directory.mkdir(parents=True,exist_ok=True)
    tag=f'{dtype}_b{batch}_{shape_mode}'
    plan=directory/(tag+'.plan');meta_path=directory/(tag+'.json')
    if plan.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text());meta['reused_plan']=True
        return plan,meta
    onnx=directory/(tag+'.onnx');export=export_onnx(model,onnx,dtype)
    logger=trt.Logger(trt.Logger.INFO);trt.init_libnvinfer_plugins(logger,'')
    builder=trt.Builder(logger);builder.max_threads=2
    network=builder.create_network(0);parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.builder_optimization_level=5
    config.avg_timing_iterations=3
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    profile=builder.create_optimization_profile()
    minimum=batch if shape_mode=='static_pad' else 1
    profile.set_shape('input',(minimum,720,32),(batch,720,32),(batch,720,32))
    config.add_optimization_profile(profile)
    cache_path=directory/(dtype+'_timing.cache')
    timing=config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'')
    config.set_timing_cache(timing,False)
    start=time.perf_counter();serialized=builder.build_serialized_network(network,config)
    if serialized is None:raise RuntimeError('TensorRT returned no engine')
    plan.write_bytes(bytes(serialized));cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    meta=dict(tensorrt=trt.__version__,dtype=dtype,batch=batch,shape_mode=shape_mode,
        min_batch=minimum,opt_batch=batch,max_batch=batch,workspace_bytes=4<<30,
        builder_optimization_level=5,avg_timing_iterations=3,tf32=False,
        export=export,build_s=time.perf_counter()-start,plan_bytes=plan.stat().st_size,
        checkpoint_sha256=hashlib.sha256(Path('artifacts/model.pt').read_bytes()).hexdigest(),reused_plan=False)
    save(meta_path,meta);return plan,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['trt','custom'],default='trt')
    p.add_argument('--variant',choices=['old','fixed','norm','new'],default='new')
    p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--shape-mode',choices=['dynamic','static_pad'],default='dynamic')
    p.add_argument('--execution',choices=['both','graph','direct'],default='both')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engine-dir',type=Path,default=Path('artifacts/trt_benchmark/engines'))
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--warmup-cycles',type=int,default=5)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
    os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING']='0'
    import numpy as np
    import torch
    from autoencoder.model import load_model
    from gpu import gpu_suite as g
    # TensorRT and every custom candidate use the same non-default stream.
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    a.output.mkdir(parents=True,exist_ok=True)
    model,checkpoint=load_model();build_meta=None
    if a.family=='trt':
        from gpu.tensorrt_backend import EngineScore
        plan,build_meta=build(model,a.engine_dir,a.dtype,a.batch,a.shape_mode)
        class StaticScore(EngineScore):
            def __call__(self,data):
                context,x,y,graph=self.entry(a.batch,720)
                x[:len(data)].copy_(data)
                if graph is not None:graph.replay()
                else:assert context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
                return y[:len(data)]
        def make(execution):
            cls=StaticScore if a.shape_mode=='static_pad' else EngineScore
            return cls(plan,graph=execution=='graph')
        executions=['graph','direct'] if a.execution=='both' else [a.execution]
        if a.batch%16:executions.reverse()
    else:
        from gpu.fused_ffn import replace_ffn
        from gpu.optimization_kernels import replace_experimental
        model=model.cuda().bfloat16()
        if a.variant=='old':model=replace_ffn(model,full=True)
        else:
            variant={'new':'norm_residual','norm':'norm','fixed':'fixed'}[a.variant]
            config=((64,64,4) if a.batch==32 else (128,64,8)) if a.variant=='new' else (128,64,8)
            model=replace_experimental(model,variant,config)
        eager=g.Score(model,torch.bfloat16).eval()
        class Compiled(torch.nn.Module):
            def __init__(self,s):
                super().__init__();self.s=torch.compile(s,mode='max-autotune',fullgraph=True,dynamic=False)
            def forward(self,x):
                torch.compiler.cudagraph_mark_step_begin();return self.s(x).clone()
        def make(_):return Compiled(eager)
        executions=['graph']
    records={}
    with torch.inference_mode():
        for execution in executions:
            s=make(execution)
            saved=np.load('artifacts/validation_reference_720.npz')
            data=torch.from_numpy(saved['input']).cuda()
            outputs=torch.cat([s(part).clone() for part in data.split(min(8,a.batch))]).cpu().numpy()
            diff=np.abs(outputs-saved['scores'])
            validation=dict(max_abs=float(diff.max()),max_relative=float((diff/np.maximum(np.abs(saved['scores']),1e-12)).max()),
                alert_agreement=float(np.mean((outputs>saved['threshold'])==(saved['scores']>saved['threshold']))))
            assert np.isfinite(outputs).all() and validation['max_relative']<=.005,validation
            folder=a.output/execution;folder.mkdir(parents=True,exist_ok=True);g.A=folder
            save(folder/'validation.json',validation)
            g.scorer=lambda _:(s,checkpoint)
            kind=f'trt_{execution}_{a.dtype}' if a.family=='trt' else f'custom_{a.variant}_bf16'
            record=g.pipeline(kind,a.batch,'gpu_normalize',3000,a.cycles,
                              warmup_cycles=a.warmup_cycles,profile=a.profile)
            assert record['cpu_score_agreement']['max_relative']<=.005,record['cpu_score_agreement']
            record['summary_ms']=stats([r['wall_s']*1000 for r in record['cycles']])
            record['configuration']=dict(family=a.family,variant=a.variant,dtype=a.dtype,batch=a.batch,
                shape_mode=a.shape_mode,execution=execution,explicit_compute_stream=True,
                full_cycle_warmups=a.warmup_cycles,build=build_meta)
            record['source_sha256']={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
                ['gpu/trt_comparison.py','gpu/tensorrt_backend.py','gpu/optimization_kernels.py','gpu/fused_ffn.py','gpu/gpu_suite.py']}
            save(folder/'result.json',record);records[execution]=record;save(a.output/'result.json',records)
            print('RESULT',kind,a.batch,a.shape_mode,json.dumps(record['summary_ms']),flush=True)
            if a.family=='trt':s.inspect(folder/'engine_inspector.json',a.batch)
            del s;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
