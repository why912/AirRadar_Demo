import torch
import torch.nn as nn

class AirEmbedding(nn.Module):
    """
    Embed categorical variables.
    Expected order of input channels: [wind_dir, weather, hour_of_day(0-23), weekday(0-6)].
    We defensively clamp indices to avoid IndexError when upstream values exceed ranges.
    """
    def __init__(self):
        super(AirEmbedding, self).__init__()
        self.embed_wdir = nn.Embedding(11, 3)      # wind direction 0..10
        self.embed_weather = nn.Embedding(18, 4)   # weather id 0..17
        self.embed_hour = nn.Embedding(24, 3)      # hour 0..23 (rename from confusing typo)
        self.embed_weekday = nn.Embedding(7, 5)    # weekday 0..6

    def forward(self, x):
        # x shape [..., 4]
        wdir = torch.clamp(x[..., 0], 0, 10)
        weather = torch.clamp(x[..., 1], 0, 17)
        hour = torch.clamp(x[..., 2], 0, 23)
        weekday = torch.clamp(x[..., 3], 0, 6)
        x_wdir = self.embed_wdir(wdir)
        x_weather = self.embed_weather(weather)
        x_hour = self.embed_hour(hour)
        x_weekday = self.embed_weekday(weekday)
        out = torch.cat((x_wdir, x_weather, x_hour, x_weekday), -1)
        return out
    
class AirEmbeddingV4(AirEmbedding):
    """Deprecated duplicate kept for backward compatibility; now inherits AirEmbedding."""
    pass
