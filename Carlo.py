# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import numpy as np
from scipy.stats import lognorm

app = Flask(__name__)

# Constantes
sims = 10000
STDEV = 3.29

# Funções de Utilidade
def check_value_not_zero(val):
    return 1 if val == 0 else val

def lognorminvpert(min_val, pert, max_val):
    mean = np.log(pert)
    sigma = (np.log(max_val) - np.log(min_val)) / STDEV
    return lognorm.ppf(np.random.rand(), s=sigma, scale=np.exp(mean))

def lognorm_risk_pert(minfreq, pertfreq, maxfreq, minloss, pertloss, maxloss):
    freq = lognorminvpert(minfreq, pertfreq, maxfreq)
    loss = lognorminvpert(minloss, pertloss, maxloss)
    return freq * loss

def generate_sim_data(rdata, totalTE, teNo):
    sim_data = np.zeros(sims)
    total_sims = sims * totalTE
    cumulative_total = sims * teNo
    
    for sim_ctr in range(sims):
        sim_data[sim_ctr] = lognorm_risk_pert(
            rdata['minfreq'], rdata['pertfreq'], rdata['maxfreq'],
            rdata['minloss'], rdata['pertloss'], rdata['maxloss']
        )
    return sim_data

def find_min(farr):
    return np.min(farr)

def bin_width(farr):
    q75, q25 = np.percentile(farr, [75 ,25])
    return 2 * (q75 - q25) * (1 / (len(farr) ** (1 / 3)))

def get_bins(sim_res, nobins):
    min_val = find_min(sim_res)
    bin_width_val = bin_width(sim_res)
    bins = np.arange(0, min_val + nobins * bin_width_val, bin_width_val)
    return bins

def put_data_in_bins(sim_res, r_bins):
    freqs, _ = np.histogram(sim_res, bins=r_bins)
    return freqs

def get_lecs(sfreqs, sumbf):
    return (sumbf - np.cumsum(sfreqs)) / sumbf

def get_cum_freqs(freqs):
    cum_freqs = np.cumsum(freqs) / np.sum(freqs)
    return cum_freqs

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    threat_events = data['threat_events']
    
    totalTE = len(threat_events)
    
    all_sim_results = []
    for teNo, rdata in enumerate(threat_events):
        sim_results = generate_sim_data(rdata, totalTE, teNo)
        all_sim_results.append(sim_results)
    
    # Agregar os riscos
    aggregated_risks = np.sum(all_sim_results, axis=0)
    
    # Calcular os bins e as frequências
    no_of_bins = int(np.ceil(np.sqrt(sims)))
    bins = get_bins(aggregated_risks, no_of_bins)
    freqs = put_data_in_bins(aggregated_risks, bins)
    
    sum_bin_freq = np.sum(freqs)
    lecs = get_lecs(freqs, sum_bin_freq)
    cum_freqs = get_cum_freqs(freqs)
    
    return jsonify({
        'bins': bins.tolist(),
        'freqs': freqs.tolist(),
        'lecs': lecs.tolist(),
        'cum_freqs': cum_freqs.tolist()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
