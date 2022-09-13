import argparse
import os.path

import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess

from . import pipeline


def predictSimple(input, model, iteration):
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB

    start_time = time.time()
    total_residues = 0
    protein_list = []
    total_step = 0
    
    dataset_valid = input["dataset_valid"]
    BATCH_COPIES = input["BATCH_COPIES"]
    device = input["device"]
    chain_id_dict = input["chain_id_dict"]
    fixed_positions_dict = input["fixed_positions_dict"]
    omit_AA_dict = input["omit_AA_dict"]
    tied_positions_dict = input["tied_positions_dict"]
    pssm_dict = input["pssm_dict"]
    bias_by_res_dict = input["bias_by_res_dict"]
    args = input["args"]
    base_folder = input["base_folder"]
    temperatures = input["temperatures"]
    NUM_BATCHES = input["NUM_BATCHES"]
    omit_AAs_np = input["omit_AAs_np"]
    bias_AAs_np = input["bias_AAs_np"]
    seed = input["seed"]

    sequences=[]
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        #print('Generating sequences...')
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            global_score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=args.ca_only)
            pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float() #1.0 for true, 0.0 for false
            name_ = batch_clones[0]['name']
            if (False):
                print("")
            else:
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask*chain_M*chain_M_pos
                scores = _scores(S, log_probs, mask_for_loss) #score only the redesigned part
                native_score = scores.cpu().data.numpy()
                global_scores = _scores(S, log_probs, mask) #score the whole structure-sequence
                global_native_score = global_scores.cpu().data.numpy()
                # Generate some sequences
                ali_file = base_folder + '/seqs/' + batch_clones[0]['name'] + '.fa'
                score_file = base_folder + '/scores/' + batch_clones[0]['name'] + '.npz'
                probs_file = base_folder + '/probs/' + batch_clones[0]['name'] + '.npz'
                print(f'Generating sequences for: {name_}')
                t0 = time.time()
                with open(ali_file, 'w') as f:
                    for temp in temperatures:
                        for j in range(NUM_BATCHES):
                            randn_2 = torch.randn(chain_M.shape, device=X.device)
                            if tied_positions_dict == None:
                                sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)
                                S_sample = sample_dict["S"] 
                            else:
                                sample_dict = model.tied_sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res_all)
                            # Compute scores
                                S_sample = sample_dict["S"]
                            log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                            mask_for_loss = mask*chain_M*chain_M_pos
                            scores = _scores(S_sample, log_probs, mask_for_loss)
                            scores = scores.cpu().data.numpy()
                            
                            global_scores = _scores(S_sample, log_probs, mask) #score the whole structure-sequence
                            global_scores = global_scores.cpu().data.numpy()
                            
                            all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                            all_log_probs_list.append(log_probs.cpu().data.numpy())
                            S_sample_list.append(S_sample.cpu().data.numpy())
                            for b_ix in range(BATCH_COPIES):
                                masked_chain_length_list = masked_chain_length_list_list[b_ix]
                                masked_list = masked_list_list[b_ix]
                                seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                                seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                                score = scores[b_ix]
                                score_list.append(score)
                                global_score = global_scores[b_ix]
                                global_score_list.append(global_score)
                                native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                                if b_ix == 0 and j==0 and temp==temperatures[0]:
                                    start = 0
                                    end = 0
                                    list_of_AAs = []
                                    for mask_l in masked_chain_length_list:
                                        end += mask_l
                                        list_of_AAs.append(native_seq[start:end])
                                        start = end
                                    native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                    l0 = 0
                                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                        l0 += mc_length
                                        native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                        l0 += 1
                                    sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                                    print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                                    sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                                    print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                                    native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                                    global_native_score_print = np.format_float_positional(np.float32(global_native_score.mean()), unique=False, precision=4)
                                    script_dir = os.path.dirname(os.path.realpath(__file__))
                                    try:
                                        commit_str = subprocess.check_output(f'git --git-dir {script_dir}/.git rev-parse HEAD', shell=True).decode().strip()
                                    except subprocess.CalledProcessError:
                                        commit_str = 'unknown'
                                    if args.ca_only:
                                        print_model_name = 'CA_model_name'
                                    else:
                                        print_model_name = 'model_name'
#                                    f.write('>{}, score={}, global_score={}, fixed_chains={}, designed_chains={}, {}={}, git_hash={}, seed={}\n{}\n'.format(name_+"_iter"+str(iteration).zfill(6), native_score_print, global_native_score_print, print_visible_chains, print_masked_chains, print_model_name, args.model_name, commit_str, seed, native_seq)) #write the native sequence
                                start = 0
                                end = 0
                                list_of_AAs = []
                                for mask_l in masked_chain_length_list:
                                    end += mask_l
                                    list_of_AAs.append(seq[start:end])
                                    start = end
    
                                seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                                l0 = 0
                                for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                    l0 += mc_length
                                    seq = seq[:l0] + '/' + seq[l0:]
                                    l0 += 1
                                sequences.append(seq)
                                score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                                global_score_print = np.format_float_positional(np.float32(global_score), unique=False, precision=4)
                                seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                                sample_number = j*BATCH_COPIES+b_ix+1
                                f.write('>T={},iter={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n'.format(temp,str(iteration).zfill(6),sample_number,score_print,global_score_print,seq_rec_print,seq)) #write generated sequence
    t1 = time.time()
    dt = round(float(t1-t0), 4)
    num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
    total_length = X.shape[1]
    print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')

    return sequences

def setPrams():    
   # from pipeline import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
  #  from pipeline import StructureDataset, StructureDatasetPDB, ProteinMPNN
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

    args = pipeline.get_args()

    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
      
    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else: 
        file_path = os.path.realpath(__file__)
        k = file_path.rfind("/")
        if args.ca_only:
            model_folder_path = file_path[:k] + '/ca_model_weights/'
        else:
            model_folder_path = file_path[:k] + '/vanilla_model_weights/'

    checkpoint_path = model_folder_path + f'{args.model_name}.pt'
    folder_for_outputs = args.out_folder
    
    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        print(40*'-')
        print('chain_id_jsonl is NOT loaded')
        
    if os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('fixed_positions_jsonl is NOT loaded')
        fixed_positions_dict = None
    
    
    if os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        print(40*'-')
        print('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    
    if os.path.isfile(args.omit_AA_jsonl):
        with open(args.omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('omit_AA_jsonl is NOT loaded')
        omit_AA_dict = None
    
    
    if os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    
    if os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        print(40*'-')
        print('tied_positions_jsonl is NOT loaded')
        tied_positions_dict = None

    
    if os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        print('bias by residue dictionary is loaded')
    else:
        print(40*'-')
        print('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   

 
    print(40*'-')
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    

    pdb_dict_list = parse_PDB(args.pdb_path, ca_only=args.ca_only)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]

    base_folder = folder_for_outputs

    #hogehoge
    input=dict()
    input["dataset_valid"] = dataset_valid
    input["BATCH_COPIES"] = BATCH_COPIES
    input["device"] = device
    input["chain_id_dict"] = chain_id_dict
    input["fixed_positions_dict"] = fixed_positions_dict
    input["omit_AA_dict"] = omit_AA_dict
    input["tied_positions_dict"] = tied_positions_dict
    input["pssm_dict"] = pssm_dict
    input["bias_by_res_dict"] = bias_by_res_dict
    input["args"] = args
    input["base_folder"] = base_folder
    input["temperatures"] = temperatures
    input["NUM_BATCHES"] = NUM_BATCHES
    input["omit_AAs_np"] = omit_AAs_np
    input["bias_AAs_np"] = bias_AAs_np
    input["seed"] = seed
    input["checkpoint_path"] = checkpoint_path
    input["folder_for_outputs"] = folder_for_outputs

    modelParams = dict()
    modelParams["ca_only"] = args.ca_only
    modelParams["num_letters"] = 21
    modelParams["hidden_dim"]=hidden_dim
    modelParams["num_layers"] = num_layers
    modelParams["augument_eps"] = args.backbone_noise

    return input, modelParams
    
                            

def main():
    
#    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
 #   from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN


    input, modelParams = setPrams()
    
    print(40*'-')
    checkpoint = torch.load(input["checkpoint_path"], map_location=input["device"]) 
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')


    
    model = pipeline.ProteinMPNN(ca_only=modelParams["ca_only"],
                        num_letters=modelParams["num_letters"],
                        node_features=modelParams["hidden_dim"],
                        edge_features=modelParams["hidden_dim"],
                        hidden_dim=modelParams["hidden_dim"],
                        num_encoder_layers = modelParams["num_layers"],
                        num_decoder_layers= modelParams["num_layers"],
                        augment_eps = modelParams["augument_eps"],
                        k_neighbors=checkpoint['num_edges'])
    model.to(input["device"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    pipeline.makeOutputPath(input["folder_for_outputs"],input["args"])    
    # Validation epoch
    iteration = 0
    sequences = predictSimple(input,model,iteration)
    print(sequences)

    
