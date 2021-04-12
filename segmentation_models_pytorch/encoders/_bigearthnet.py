""" BigEarthNet pretrained encoders URL-s
"""

_url_format = 'https://artifacts.stageogip.ru/artifactory/ogip-image-processing/models/bigearthnet/{}-{}.pth'

_model_uuid = {
    'resnest101': 'c3647638',
}

def _short_hash(name):
    if name not in _model_uuid:
        raise ValueError('BigEarth pretrained model for {name} is not available.'.format(name=name))
    return _model_uuid[name][:8]

bigearthnet_urls = {
    name: _url_format.format(name, _short_hash(name)) for name in _model_uuid.keys()
}
