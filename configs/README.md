# Configuration layout

配置按 `benchmark_spec.yaml`、模型、环境、方法和实验切分。公开配置只包含公开规则与 opaque split ID，不包含 key、hidden evaluator、private seed 或完整任务真值。

每份实验配置应包含：`experiment_id`、`version`、`status`、task family/split、protocol/baseline、agent count、model/provider/version、world resources `R`、system budget `B`、individual utility、communication/authority policy、evaluator version、seed axes、repetitions、cost cap、expected artifacts 和 freeze commit。

