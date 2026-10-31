# Deployment entry / 部署入口

Use the maintained language-specific guides; this compatibility entry intentionally does not
duplicate startup commands, permission lists or credential instructions.

请使用下面的维护版指南；本兼容入口不再重复启动命令、权限清单或凭据步骤，以免版本漂移。

- [中文部署指南](DEPLOYMENT.zh-CN.md) / [English deployment](DEPLOYMENT.en.md)
- [API 与密钥配置](CONFIGURATION.zh-CN.md) / [API and key setup](CONFIGURATION.en.md)
- [Discord Bot 配置 / Setup](DISCORD_SETUP.md)
- [安全与提交检查 / Security](../SECURITY.md)
- [升级回滚 / Rollback](ROLLBACK.md)

Windows DPAPI credentials require the maintained launcher. Bare Docker Compose needs separately
provided credentials. Never overwrite an existing `.env`, regenerate its data-encryption key or
interrupt an active meeting merely to follow a first-install example. Documentation changes alone
do not require rebuilding or deploying the current installation.

Windows DPAPI 密钥必须由维护的启动器读取；手动 Compose 需另行提供凭据。不要为照抄首次安装
示例而覆盖已有 `.env`、重新生成旧数据的加密密钥或中断正在进行的会议。仅文档改动不要求
重建或部署本机现有版本。
