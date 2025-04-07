
# pipeline/config.py

LABEL_MAPPING = {
    "BenignTraffic": 0,
    "BruteForce": 1,
    "DDoS": 2,
    "DoS": 3,
    "Mirai": 4,
    "Recon": 5,
    "Spoofing": 6,
    "Web-Based": 7,
}

# Categorical & numerical feature lists
categorical_columns = ['stream', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip', 'port_class_dst', 'l4_tcp',
                       'l4_udp', 'handshake_version', 'handshake_ciphersuites', 'handshake_sig_hash_alg_len',
                       'tls_server', 'http_request_method', 'http_host', 'user_agent', 'dns_server', 'dns_query_type',
                       'device_mac', 'eth_src_oui', 'eth_dst_oui', 'http_content_type', 'icmp_type',
                       'icmp_checksum_status', 'http_uri']


numerical_columns = [
    'inter_arrival_time', 'time_since_previously_displayed_frame', 'ttl', 'eth_size', 'tcp_window_size',
    'payload_entropy', 'handshake_cipher_suites_length', 'handshake_extensions_length', 'dns_len_qry',
    'dns_interval', 'dns_len_ans', 'payload_length', 'http_content_len', 'icmp_data_size',
    'jitter', 'stream_1_count', 'stream_1_mean', 'stream_1_var', 'src_ip_1_count', 'src_ip_1_mean', 'src_ip_1_var',
    'src_ip_mac_1_count', 'src_ip_mac_1_mean', 'src_ip_mac_1_var', 'channel_1_count', 'channel_1_mean', 'channel_1_var',
    'stream_jitter_1_sum', 'stream_jitter_1_mean', 'stream_jitter_1_var', 'stream_5_count', 'stream_5_mean',
    'stream_5_var', 'src_ip_5_count', 'src_ip_5_mean', 'src_ip_5_var', 'src_ip_mac_5_count', 'src_ip_mac_5_mean',
    'src_ip_mac_5_var', 'channel_5_count', 'channel_5_mean', 'channel_5_var', 'stream_jitter_5_sum',
    'stream_jitter_5_mean', 'stream_jitter_5_var', 'stream_10_count', 'stream_10_mean', 'stream_10_var',
    'src_ip_10_count', 'src_ip_10_mean', 'src_ip_10_var', 'src_ip_mac_10_count', 'src_ip_mac_10_mean',
    'src_ip_mac_10_var', 'channel_10_count', 'channel_10_mean', 'channel_10_var', 'stream_jitter_10_sum',
    'stream_jitter_10_mean', 'stream_jitter_10_var', 'stream_30_count', 'stream_30_mean', 'stream_30_var',
    'src_ip_30_count', 'src_ip_30_mean', 'src_ip_30_var', 'src_ip_mac_30_count', 'src_ip_mac_30_mean',
    'src_ip_mac_30_var', 'channel_30_count', 'channel_30_mean', 'channel_30_var', 'stream_jitter_30_sum',
    'stream_jitter_30_mean', 'stream_jitter_30_var', 'stream_60_count', 'stream_60_mean', 'stream_60_var',
    'src_ip_60_count', 'src_ip_60_mean', 'src_ip_60_var', 'src_ip_mac_60_count', 'src_ip_mac_60_mean',
    'src_ip_mac_60_var', 'channel_60_count', 'channel_60_mean', 'channel_60_var', 'stream_jitter_60_sum',
    'stream_jitter_60_mean', 'stream_jitter_60_var', 'ntp_interval', 'most_freq_spot', 'min_et', 'q1', 'min_e',
    'var_e', 'q1_e', 'sum_p', 'min_p', 'max_p', 'med_p', 'average_p', 'var_p', 'q3_p', 'q1_p', 'iqr_p', 'l3_ip_dst_count'
]