import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead
from monai.networks import nets

class ResnetEmbeddingModel(nn.Module):
    """
    Embedding model for imaging trained with 3D ResNet backbone.
    """
    def __init__(self, args) -> None:
        super(ResnetEmbeddingModel, self).__init__()

        self.keep_projector = args.keep_projector

        # Load weights
        checkpoint = torch.load(args.checkpoint)
        original_args = checkpoint['hyper_parameters']
        state_dict = checkpoint['state_dict']

        # Load architecture
        if original_args['model'] == 'resnet18':
            model = nets.resnet18(spatial_dims=3, n_input_channels=1, pretrained=False)
            pooled_dim = 512
        elif original_args['model'] == 'resnet50':
            model = nets.resnet50(spatial_dims=3, n_input_channels=1, pretrained=False)
            pooled_dim = 2048
        else:
            raise Exception('Invalid architecture. Please select either resnet18 or resnet50.')

        self.backbone = nn.Sequential(*list(model.children())[:-1]) 

        self.projection_head = SimCLRProjectionHead(pooled_dim, original_args['embedding_dim'], original_args['projection_dim'])

        # Remove prefix and fc layers
        state_dict_encoder = {}
        state_dict_projector = {}
        for k in list(state_dict.keys()):
            if k.startswith('encoder_imaging.'):
                state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
            if k.startswith('projection_head_imaging.'):
                state_dict_projector[k[len('projection_head_imaging.'):]] = state_dict[k]

        log = self.backbone.load_state_dict(state_dict_encoder, strict=True)
        assert len(log.missing_keys) == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("forward")
        embeddings = self.backbone(x).squeeze()
        print("Embeddings shape", embeddings.shape)

        if self.keep_projector:
            embeddings = self.projection_head(embeddings)

        return embeddings
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, last_bn=True):
        super(ProjectionHead, self).__init__()

        self.last_bn = last_bn
        
        # Projection head with one hidden layer of size 2048
        layers = [
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False)
        ]
        
        self.projection_head = nn.Sequential(*layers)
        
        # Normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(output_dim, affine=False)
        
    def forward(self, x):
        out = self.projection_head(x)
        if self.last_bn:
            out = self.bn(out)
        return out
