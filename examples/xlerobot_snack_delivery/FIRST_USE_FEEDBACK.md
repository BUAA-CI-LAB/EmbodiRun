# XLeRobot first-use trial and feedback

这份材料用于课堂试用或新用户第一次上手。当前提供的是流程和记录模板，
尚未收集真实同学反馈。请在可使用的 Linux 主机上只做无设备软件演练。

## 试用流程

1. 从 [README](README.md) 选择 native 或 Docker，记录开始时间、OS/架构和版本。
   首次安装时记录下载缓存是否为空；不要清理共享缓存来制造首次安装。
2. 依次完成 `init → validate → plan → setup --mode software → check --mode software --json → dry-run`。
   Docker 使用 `xlerobot-software`，镜像已有环境，跳过 `setup`。
   遇到已有配置就复用它；native 转 Docker 前先备份 manifest，只去掉 Docker 副本的
   顶层 `python` override，保留其他字段和标定，或另建 Docker 目录再迁配置。
   不确定时记录问题和第一条阻塞错误。
3. Recipe CLI 打印 `Outputs:` 后，记录其中的 `run.json`、`command-0.log`；
   dry-run 执行后再查看 `result/status.json`。
   若 `uv sync` / `docker compose build` 在 CLI 前失败，尚无这些文件，
   保存该命令的终端 stdout/stderr、完整命令、退出码和所选日志位置即可。
   说明哪一步成功、哪一步仍未完成。可把 [Agent 起始 prompt](AGENT_GUIDE.md)
   交给自己的 coding Agent，再记录它问了什么、做了什么、在哪一步需要人帮助。
4. 填写下面的短表并记录结束时间。软件演练到此结束，不启动 owner/Control，
   不运行 `up` 或 `run --allow-hardware`。

## 记录模板

```text
匿名试用编号：
日期、checkout 分支/commit：
目标主机 OS / 架构：
native 或 Docker；Python/uv 或 Docker/Compose 版本：
此前是否使用过 EmbodiRun / XLeRobot：
独立照 README / Agent 协助 / 同伴协助：
开始时间 / 结束时间 / 下载缓存状态：
实际命令与退出码：
到达的最后一步：
check 的 status 与首条 issue（code / location / next_action）：
第一条阻塞错误及日志位置：
重试次数、解决方式和外部帮助：
bootstrap/build 终端日志，或 run.json / result/status.json 的位置和结果（未生成则写未生成）：
尚未完成或仍不理解的部分：
```

可附经过脱敏的日志片段。不要提交认证文件、配对码、token、私人 IP/设备标识、
真实路线/室内地图或含个人信息的画面。公开反馈使用匿名编号与必要的错误信息。

## 试用后问五个问题

- 你在哪里找到入口，能否判断该选软件演练还是准备小车，哪些前提现在不用准备？
- 你能否说清 `deployment.local.yaml`、`hardware.local.json`、`config.local.json`
  分别改什么；哪一处需要更具体的例子？
- 看过结果文件后，软件成功说明了什么；为什么此时还不能据此上真机？
- 你在哪里需要问人或缺少路径，`check --json` 的 `next_action` 能否帮助你继续？
- 若使用 Agent，它是否复用了当前目录、只问必要信息、清楚报告范围；哪一步仍需要人工帮助？

整理反馈时区分首次成功、重试成功、需要协助和未完成，保留原始问题。
课堂软件试用和 Agent 辅助记录不代表完整冷安装、模型质量或物理递送验证。
