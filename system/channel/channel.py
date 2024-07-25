import matplotlib.pyplot as plt
import numpy as np
from numpy import round
import pickle
import time
import tensorflow as tf
import random
import torch
from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber, PlotBER
from sionna.utils.metrics import compute_ber
from decimal import Decimal, ROUND_HALF_UP
from utils.data_utils  import calculate_data_rate, calculate_upload_time


def cdl_channel_user(client):

    #np.random.seed(client)
    random.seed(client)
    # Define the number of UT and BS antennas.
    # For the CDL model, that will be used in this notebook, only
    # a single UT and BS are supported.
    
    #num_ut = 1
    #num_bs = 1
    num_ut_ant = 1
    num_bs_ant = 1

    # The number of transmitted streams is equal to the number of UT antennas
    # in both uplink and downlink
    num_streams_per_tx = num_ut_ant

    # Create an RX-TX association matrix
    rx_tx_association = np.array([[1]])

    sm = StreamManagement(rx_tx_association, num_streams_per_tx)


    rg = ResourceGrid(num_ofdm_symbols=14,
                    fft_size=76,
                    subcarrier_spacing=15e3,
                    num_tx=1,
                    num_streams_per_tx=num_streams_per_tx,
                    cyclic_prefix_length=6,
                    num_guard_carriers=[5,6],
                    dc_null=True,
                    pilot_pattern="kronecker",
                    pilot_ofdm_symbol_indices=[2,11])

    carrier_frequency = 2.6e6 # Carrier frequency in Hz.
                            # This is needed here to define the antenna element spacing.

    ut_array = AntennaArray(num_rows=1,
                            num_cols=int(num_ut_ant/2),
                            polarization="dual",
                            polarization_type="cross",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=int(num_bs_ant/2),
                            polarization="dual",
                            polarization_type="cross",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                        # In the `uplink`, the UT is transmitting.

    lower_bound_ns = 65e-9  # 45 ns
    upper_bound_ns = 634e-9  # 316 ns

    # Gerando um número aleatório entre 65 ns e 634 ns
    delay_spread = np.random.uniform(lower_bound_ns, upper_bound_ns)


    cdl_model = "B"       # Suitable values are ["A", "B", "C", "D", "E"]
 
    max_speed = np.random.uniform(low=0, high=0)  # UT speed [m/s].
    min_speed = np.random.uniform(low=0, high=max_speed)  # UT speed [m/s].

    cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=min_speed, max_speed=max_speed)

    l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
    l_tot = l_max-l_min+1

    num_bits_per_symbol = 2 # QPSK modulation
    coderate = 0.5 # Code rate
    n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
    k = int(n*coderate) # Number of information bits

    # The binary source will create batches of information bits
    binary_source = BinarySource()

    # The encoder maps information bits to coded bits
    encoder = LDPC5GEncoder(k, n)

    # The mapper maps blocks of information bits to constellation symbols
    mapper = Mapper("qam", num_bits_per_symbol)

    # The resource grid mapper maps symbols onto an OFDM resource grid
    rg_mapper = ResourceGridMapper(rg)

    # OFDM modulator and demodulator
    modulator = OFDMModulator(rg.cyclic_prefix_length)
    demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)

    # The LS channel estimator will provide channel estimates and error variances
    ls_est = LSChannelEstimator(rg, interpolation_type="nn")

    # The LMMSE equalizer will provide soft symbols together with noise variance estimates
    lmmse_equ = LMMSEEqualizer(rg, sm)

    # The demapper produces LLR for all coded bits
    demapper = Demapper("app", "qam", num_bits_per_symbol)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    batch_size = 100 # We pick a small batch_size as executing this code in Eager mode could consume a lot of memory
    #ebno_db = random.uniform(10, 22) #The `Eb/No` value in dB
    ebno_db = np.random.uniform(low=7, high=17)
    #ebno_db = 7
    #Computes the Noise Variance (No)
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
    c = encoder(b)
    x = mapper(c)
    x_rg = rg_mapper(x)

 
    cir = cdl(batch_size, rg.num_time_samples+l_tot-1, rg.bandwidth)

    # OFDM modulation with cyclic prefix insertion
    x_time = modulator(x_rg)

    # Compute the discrete-time channel impulse reponse
    h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min, l_max, normalize=True)

    time_channel = TimeChannel(cdl, rg.bandwidth, rg.num_time_samples,
                               l_min=l_min, l_max=l_max, normalize_channel=True,
                               add_awgn=True, return_channel=True)

    y_time, h_time = time_channel([x_time, no])
    # Compute the channel output

    # OFDM demodulation and cyclic prefix removal
    y = demodulator(y_time)

    h_hat, err_var = ls_est ([y, no])

    x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
    llr = demapper([x_hat, no_eff])
    b_hat = decoder(llr)
    ber = compute_ber(b, b_hat)
    
    data_rate = calculate_data_rate(bandwidth = rg.bandwidth , snr_db = ebno_db , num_bits_per_symbol = num_bits_per_symbol, coderate = coderate)
    no = no.numpy().item()
    return ebno_db, no, data_rate, rg.bandwidth

ebno_db, no , data_rate, bandwidth = cdl_channel_user(2)
print(f'ebno_db: {ebno_db}   no: {no}  bandwidth:{bandwidth}')