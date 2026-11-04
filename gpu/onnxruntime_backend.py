"""CUDA-only ONNX Runtime with external-stream I/O binding and optional graphs."""
from pathlib import Path
import json
import numpy as np
import torch

class OnnxScore:
    def __init__(self,path,graph=True,profile_dir=None,sdpa_kernel=0):
        import onnxruntime as ort
        ort.preload_dlls()
        self.ort=ort;self.graph=graph;self.entries={};self.profile=profile_dir is not None
        options=ort.SessionOptions()
        options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        if self.profile:
            options.enable_profiling=True
            options.profile_file_prefix=str(Path(profile_dir)/'ort_nodes')
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=[('CUDAExecutionProvider',{
            'device_id':0,'user_compute_stream':str(torch.cuda.current_stream().cuda_stream),
            'use_tf32':'0','enable_cuda_graph':'1' if graph else '0','gpu_mem_limit':str(4<<30),
            'sdpa_kernel':str(sdpa_kernel)})])
        assert 'CUDAExecutionProvider' in self.session.get_providers()
    def entry(self,n,length):
        key=(n,length)
        if key not in self.entries:
            x=torch.empty(n,length,32,device='cuda',dtype=torch.float32)
            y=torch.empty(n,device='cuda',dtype=torch.float32)
            binding=self.session.io_binding()
            binding.bind_input('input','cuda',0,np.float32,tuple(x.shape),x.data_ptr())
            binding.bind_output('scores','cuda',0,np.float32,tuple(y.shape),y.data_ptr())
            options=self.ort.RunOptions()
            options.add_run_config_entry('disable_synchronize_execution_providers','1')
            if self.graph:options.add_run_config_entry('gpu_graph_id',str(len(self.entries)))
            self.entries[key]=(x,y,binding,options)
        return self.entries[key]
    def __call__(self,data):
        x,y,binding,options=self.entry(len(data),data.shape[1])
        x.copy_(data)
        self.session.run_with_iobinding(binding,options)
        return y
    def inspect(self,path,batch,length=720):
        Path(path).write_text(json.dumps(dict(onnxruntime=self.ort.__version__,
            providers=self.session.get_providers(),provider_options=self.session.get_provider_options(),
            graph=self.graph,cpu_fallback_disabled=True,tf32=False,batch=batch,length=length),indent=2))
    def __del__(self):
        try:
            if self.profile:self.session.end_profiling()
        except Exception:pass
