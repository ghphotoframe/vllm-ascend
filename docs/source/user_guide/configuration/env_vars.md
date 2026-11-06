# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

For the optional A5-only Ling prefill backend, see [FlashKDA](../feature_guide/flash_kda.md).

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}
