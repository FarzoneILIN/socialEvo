# Data layout

`public/` 保存可公开的任务模板和小型 fixture；`generated/` 保存可由 seed 重建的索引；`schemas/` 保存数据格式。private 数据不放在仓库内，由 `SOCIALEVO_PRIVATE_ROOT` 指向受控目录。

仓库只提交 public manifest、opaque private split ID、版本和 hash。原始 private 轨迹、hidden labels、完整 prompt/response 和大文件走外部 artifact storage。

