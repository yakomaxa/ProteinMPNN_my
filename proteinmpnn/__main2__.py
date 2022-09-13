import torch
from . import pmpnnPipeline                            

def main():
    
    input, modelParams = pmpnnPipeline.setPrams()
    
    print(40*'-')
    checkpoint = torch.load(input["checkpoint_path"], map_location=input["device"]) 
    print('Number of edges:', checkpoint['num_edges'])
    noise_level_print = checkpoint['noise_level']
    print(f'Training noise level: {noise_level_print}A')
    
    model = pmpnnPipeline.ProteinMPNN(ca_only=modelParams["ca_only"],
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
    pmpnnPipeline.makeOutputPath(input["folder_for_outputs"],input["args"])    
    # Validation epoch
    iteration = 0
    sequences = pmpnnPipeline.predictSimple(input,model,iteration)
    print(sequences)

    
