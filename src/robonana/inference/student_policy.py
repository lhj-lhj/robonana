"""中文：显式启用的蒸馏评测适配；不在 MAC trainer 中导入。
English: Evaluation-only student adapter. Teacher preprocessing/world stays shared.
"""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from .batched_policy import BatchedRoboNanaRobotWinPolicy
from .robotwin_policy import InferenceMode, seeded_randn_like
from robonana.models.flux2_action_student import build_action_student
from robonana.sampling import prefill_mac_condition
from robonana.inference_contract import sha256_file


class StudentRobotWinPolicy(BatchedRoboNanaRobotWinPolicy):
    def __init__(self, *, student_checkpoint, **kwargs):
        super().__init__(**kwargs)
        if self.inference_mode is not InferenceMode.ACTION_ONLY:
            raise ValueError('student experiment currently supports action-only evaluation')
        path=Path(student_checkpoint)
        config=json.loads((path.parent.parent/'config.json').read_text())
        if sha256_file(kwargs['checkpoint']) != config['teacher_sha256']:
            raise ValueError('student was distilled against a different teacher')
        self.student=build_action_student(self.model,config['hidden_dim']).to(self.model_device)
        self.student.load_state_dict(load_file(str(path)),strict=True)
        self.student.eval().requires_grad_(False)

    @torch.inference_mode()
    def _sample_action_batch(self, *, context,context_mask,current,state,sampling_seeds):
        self._last_batch_rejection=None
        batch=state.shape[0]
        template=torch.zeros(batch,48,self.action_dim,device=self.model_device,dtype=self.dtype)
        noise=torch.cat([seeded_randn_like(template[i:i+1],seed) for i,seed in enumerate(sampling_seeds)])
        cache=prefill_mac_condition(model=self.model,context=context,context_mask=context_mask,
            current_latents=current,state=state,grid_height=self.grid_height,grid_width=self.grid_width)
        ids=self.model._robot_ids(batch_size=batch,length=48,segment_id=3,device=self.model_device,
            dtype=torch.long,time_ids=torch.arange(1,49,device=self.model_device)[None].expand(batch,-1))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.student(cache,noise=noise,query_pe=self.model.pe_embedder(ids))
