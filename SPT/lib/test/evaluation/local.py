from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.prj_dir = '/root/autodl-tmp/UniMod1K/SPT/'
    settings.save_dir = '/root/autodl-tmp/UniMod1K/SPT/'
    settings.network_path = '/root/autodl-tmp/UniMod1K/SPT/lib/test/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/root/autodl-tmp/UniMod1K/SPT/lib/test/result_plots/'
    settings.results_path = '/root/autodl-tmp/UniMod1K/SPT/lib/test/tracking_results/'    # Where to store tracking results
    settings.unimod1k_path = '/root/autodl-tmp/1-训练验证集/ValidationSet'
    settings.segmentation_path = '/root/autodl-tmp/UniMod1K/SPT/lib/test/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

