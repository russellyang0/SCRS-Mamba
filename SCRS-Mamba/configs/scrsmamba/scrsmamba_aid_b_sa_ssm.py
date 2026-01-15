_base_ = [
    'scrsmamba_aid_b.py',
]

work_dir = 'work_dirs/scrsmamba_aid_b_sa_ssm'

model = dict(
    backbone=dict(
        path_type='sa_ssm_forward_reverse_shuffle_gate',
    ),
)
