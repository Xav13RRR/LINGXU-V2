# Deployment entry / 部署入口

Choose a deployment guide and complete API and Discord configuration before starting services.
请选择部署指南，完成 API 和 Discord 配置后再启动服务。

- [中文部署指南](DEPLOYMENT.zh-CN.md) / [English deployment](DEPLOYMENT.en.md)
- [API 与密钥配置](CONFIGURATION.zh-CN.md) / [API and key setup](CONFIGURATION.en.md)
- [Discord Bot 配置 / Setup](DISCORD_SETUP.md)
- [安全与提交检查 / Security](../SECURITY.md)
- [升级回滚 / Rollback](ROLLBACK.md)

Windows DPAPI credentials require the maintained launcher. Bare Docker Compose needs separately
provided credentials. Never overwrite an existing `.env`, regenerate its data-encryption key or
interrupt an active meeting to follow a first-install example.

Windows DPAPI 密钥必须由维护的启动器读取；手动 Compose 需另行提供凭据。不要为照抄首次安装
示例而覆盖已有 `.env`、重新生成旧数据的加密密钥或中断正在进行的会议。
