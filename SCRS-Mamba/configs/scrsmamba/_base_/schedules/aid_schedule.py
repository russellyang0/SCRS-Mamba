optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.05),
)

warmup_epochs = 10
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs),
]

train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=20)
val_cfg = dict()
test_cfg = dict()
