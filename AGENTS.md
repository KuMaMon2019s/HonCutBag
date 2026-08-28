# HonCut Codex 项目指令

## 架构事实源

- 诊断、修复或重构 HonCut 代码前，必须完整阅读 `docs/HONCUT_ARCHITECTURE.md`。
- 该文档是当前生产架构、模块所有权、恢复语义和验收边界的规范事实源；历史 roadmap、audit、redesign 文档只作背景参考。
- 如果实现与规范冲突，先追踪实际生产执行路径。若确需改变架构、公共接口、schema、恢复优先级或 owner，必须在同一提交更新架构文档和对应特征测试。

## 修复约束

- 先定位唯一 owner，再做最小修复。依赖方向保持为：CLI → Lifecycle → Graph composition → Graph node → Phase/domain → Runtime policy/executor → Provider/client 与 Artifact storage。
- `pipeline/src/phases/pipeline_core.py` 仅是测试兼容门面；禁止向其中加入生产业务、拓扑、恢复、Provider 或重试逻辑，也禁止新增生产引用。
- Graph node 只读取 State、调用一个窄 owner 并返回 canonical State patch；媒体 I/O、subprocess、模型和网络调用留在 Phase/domain 或 Runtime。
- Graph 与顺序执行器共享 Phase owner。修复公共行为时必须验证两条执行路径，禁止以静默 fallback 重复副作用。
- State 必须 JSON-safe；不得保存媒体、base64、日志正文、客户端、句柄或秘密。旧别名只在迁移适配器读取，生产代码只写 canonical 字段。
- 未知未来 State、checkpoint、task DB 或 Artifact schema，以及 run/project/hash/lineage 不匹配，必须 fail closed。
- 所有可能付费的长视频任务必须经过 `GenerationTaskStore`；`submission_uncertain` 只能恢复轮询或人工裁决，禁止盲目重提。
- timeout、retry、cooldown、backoff 和 capacity policy 只归 Runtime；不得在 Graph、Phase 或 Provider client 叠加重试。
- QA、补拍和连续性闭环必须有限且有终止状态；缺失、损坏或不可验证的 receipt/Artifact 不能解释为成功。
- Dry-run 和离线 fixture 必须明确标注 test-only，且只能通过私有依赖注入启用；不得增加普通 CLI 可误用的假 Provider 开关，也不得伪造生产 QA/Provider 凭证。
- 默认测试严禁真实付费请求。真实 Provider 提交必须获得用户当次明确的费用授权。

## 修复工作流与验收

- 除非用户另有分支要求，代码修复从干净基线创建 `codex/<scope>-fix` 分支；记录首个失败签名、输入/产物血缘、任务数量和 Provider 请求数。
- 先用回归测试冻结正确行为，再修复唯一 owner。不同 owner 或后续 Phase 的阻塞应单独记录、分支和提交，不混入当前修复。
- 每个独立行为单独提交并可独立回滚。提交前依次运行目标 pytest、`make lint`、`git diff --check` 和 `make test`。
- 涉及恢复、Provider、Artifact 或媒体时，按架构文档执行相应零请求离线验收；冷启动与恢复必须保持任务 ID、任务数量和产物哈希一致，Provider 请求数为零。
- Phase 1～Phase 9 的每个独立修复都采用持久化双门验收，且两门都通过才可宣告验收成功：`regression` 门是默认零付费的目标回归、完整测试和相应离线验收；`live_paid_provider` 门是与本次修复路径直接相关、经用户当次费用授权的一次真实付费 Provider 接口验收。真实门必须先运行无 `--submit` 预检，再用对应 Phase 的专用 live acceptance；在 Runtime 和传输边界硬限制为最多一次，调用前原子写入 `submission_uncertain` 收据，失败或中断后禁止自动重试。缺少授权时标记 `pending_live_acceptance`，真实门失败时标记 `live_acceptance_failed`，两者都不得写成验收成功；模型业务 verdict 与调用链验收结果分别记录。
