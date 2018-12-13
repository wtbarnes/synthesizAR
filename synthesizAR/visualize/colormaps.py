"""
Implement some IDL colormaps for ease of comparison
See https://github.com/planetarymike/IDL-Colorbars
"""
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['bgry_004_idl_cmap']


bgry_004_idl_cmap = LinearSegmentedColormap.from_list(
    'bgry_004_idl_cmap',
    [[0., 0., 0.],
     [0., 0., 0.00784314],
     [0., 0., 0.0156863],
     [0., 0., 0.0235294],
     [0., 0., 0.0313725],
     [0., 0., 0.0392157],
     [0., 0., 0.0470588],
     [0., 0., 0.054902],
     [0., 0., 0.0627451],
     [0., 0., 0.0705882],
     [0., 0., 0.0784314],
     [0., 0., 0.0862745],
     [0., 0., 0.0980392],
     [0., 0., 0.105882],
     [0., 0., 0.113725],
     [0., 0., 0.121569],
     [0., 0., 0.129412],
     [0., 0., 0.137255],
     [0., 0., 0.145098],
     [0., 0., 0.152941],
     [0., 0., 0.160784],
     [0., 0., 0.168627],
     [0., 0., 0.176471],
     [0., 0., 0.184314],
     [0., 0., 0.196078],
     [0., 0., 0.203922],
     [0., 0., 0.211765],
     [0., 0., 0.219608],
     [0., 0., 0.227451],
     [0., 0., 0.235294],
     [0., 0., 0.243137],
     [0., 0., 0.25098],
     [0., 0., 0.258824],
     [0., 0.0117647, 0.266667],
     [0., 0.0235294, 0.27451],
     [0., 0.0352941, 0.282353],
     [0., 0.0470588, 0.294118],
     [0., 0.0588235, 0.301961],
     [0., 0.0705882, 0.309804],
     [0., 0.0823529, 0.317647],
     [0., 0.0980392, 0.32549],
     [0., 0.109804, 0.333333],
     [0., 0.121569, 0.341176],
     [0., 0.133333, 0.34902],
     [0., 0.145098, 0.356863],
     [0., 0.156863, 0.364706],
     [0., 0.168627, 0.372549],
     [0., 0.180392, 0.380392],
     [0., 0.196078, 0.392157],
     [0., 0.207843, 0.392157],
     [0., 0.219608, 0.392157],
     [0., 0.231373, 0.392157],
     [0., 0.243137, 0.392157],
     [0., 0.254902, 0.392157],
     [0., 0.266667, 0.392157],
     [0., 0.278431, 0.392157],
     [0., 0.294118, 0.392157],
     [0., 0.305882, 0.392157],
     [0., 0.317647, 0.392157],
     [0., 0.329412, 0.392157],
     [0., 0.341176, 0.392157],
     [0., 0.352941, 0.392157],
     [0., 0.364706, 0.392157],
     [0., 0.376471, 0.392157],
     [0., 0.392157, 0.392157],
     [0., 0.403922, 0.392157],
     [0., 0.415686, 0.392157],
     [0., 0.427451, 0.392157],
     [0., 0.439216, 0.392157],
     [0., 0.45098, 0.392157],
     [0., 0.462745, 0.392157],
     [0., 0.47451, 0.392157],
     [0., 0.490196, 0.392157],
     [0., 0.501961, 0.392157],
     [0., 0.513725, 0.392157],
     [0., 0.52549, 0.392157],
     [0., 0.537255, 0.392157],
     [0., 0.54902, 0.392157],
     [0., 0.560784, 0.392157],
     [0., 0.572549, 0.392157],
     [0., 0.588235, 0.392157],
     [0., 0.588235, 0.376471],
     [0., 0.588235, 0.364706],
     [0., 0.588235, 0.352941],
     [0., 0.588235, 0.341176],
     [0., 0.588235, 0.329412],
     [0., 0.588235, 0.317647],
     [0., 0.588235, 0.305882],
     [0., 0.588235, 0.294118],
     [0., 0.588235, 0.278431],
     [0., 0.588235, 0.266667],
     [0., 0.588235, 0.254902],
     [0., 0.588235, 0.243137],
     [0., 0.588235, 0.231373],
     [0., 0.588235, 0.219608],
     [0., 0.588235, 0.207843],
     [0., 0.588235, 0.196078],
     [0., 0.584314, 0.180392],
     [0., 0.580392, 0.168627],
     [0., 0.580392, 0.156863],
     [0., 0.576471, 0.145098],
     [0., 0.572549, 0.133333],
     [0., 0.572549, 0.121569],
     [0., 0.568627, 0.109804],
     [0., 0.568627, 0.0980392],
     [0., 0.564706, 0.0823529],
     [0., 0.560784, 0.0705882],
     [0., 0.560784, 0.0588235],
     [0., 0.556863, 0.0470588],
     [0., 0.552941, 0.0352941],
     [0., 0.552941, 0.0235294],
     [0., 0.54902, 0.0117647],
     [0., 0.54902, 0.],
     [0.027451, 0.537255, 0.],
     [0.0588235, 0.529412, 0.],
     [0.0862745, 0.517647, 0.],
     [0.117647, 0.509804, 0.],
     [0.145098, 0.498039, 0.],
     [0.176471, 0.490196, 0.],
     [0.203922, 0.478431, 0.],
     [0.235294, 0.470588, 0.],
     [0.262745, 0.458824, 0.],
     [0.294118, 0.45098, 0.],
     [0.321569, 0.439216, 0.],
     [0.352941, 0.431373, 0.],
     [0.380392, 0.419608, 0.],
     [0.411765, 0.411765, 0.],
     [0.439216, 0.4, 0.],
     [0.470588, 0.392157, 0.],
     [0.490196, 0.364706, 0.],
     [0.509804, 0.341176, 0.],
     [0.529412, 0.317647, 0.],
     [0.54902, 0.294118, 0.],
     [0.568627, 0.266667, 0.],
     [0.588235, 0.243137, 0.],
     [0.607843, 0.219608, 0.],
     [0.627451, 0.196078, 0.],
     [0.647059, 0.168627, 0.],
     [0.666667, 0.145098, 0.],
     [0.686275, 0.121569, 0.],
     [0.705882, 0.0980392, 0.],
     [0.72549, 0.0705882, 0.],
     [0.745098, 0.0470588, 0.],
     [0.764706, 0.0235294, 0.],
     [0.784314, 0., 0.],
     [0.784314, 0.00784314, 0.],
     [0.788235, 0.0156863, 0.],
     [0.788235, 0.0235294, 0.],
     [0.792157, 0.0352941, 0.],
     [0.792157, 0.0431373, 0.],
     [0.796078, 0.0509804, 0.],
     [0.796078, 0.0627451, 0.],
     [0.8, 0.0705882, 0.],
     [0.8, 0.0784314, 0.],
     [0.803922, 0.0901961, 0.],
     [0.803922, 0.0980392, 0.],
     [0.807843, 0.105882, 0.],
     [0.807843, 0.113725, 0.],
     [0.811765, 0.12549, 0.],
     [0.811765, 0.133333, 0.],
     [0.815686, 0.141176, 0.],
     [0.815686, 0.152941, 0.],
     [0.819608, 0.160784, 0.],
     [0.819608, 0.168627, 0.],
     [0.823529, 0.180392, 0.],
     [0.823529, 0.188235, 0.],
     [0.827451, 0.196078, 0.],
     [0.827451, 0.207843, 0.],
     [0.831373, 0.215686, 0.],
     [0.831373, 0.223529, 0.],
     [0.835294, 0.231373, 0.],
     [0.835294, 0.243137, 0.],
     [0.839216, 0.25098, 0.],
     [0.839216, 0.258824, 0.],
     [0.843137, 0.270588, 0.],
     [0.843137, 0.278431, 0.],
     [0.847059, 0.286275, 0.],
     [0.847059, 0.298039, 0.],
     [0.85098, 0.305882, 0.],
     [0.85098, 0.313725, 0.],
     [0.854902, 0.32549, 0.],
     [0.854902, 0.333333, 0.],
     [0.858824, 0.341176, 0.],
     [0.858824, 0.34902, 0.],
     [0.862745, 0.360784, 0.],
     [0.862745, 0.368627, 0.],
     [0.866667, 0.376471, 0.],
     [0.866667, 0.388235, 0.],
     [0.870588, 0.396078, 0.],
     [0.870588, 0.403922, 0.],
     [0.87451, 0.415686, 0.],
     [0.87451, 0.423529, 0.],
     [0.878431, 0.431373, 0.],
     [0.878431, 0.443137, 0.],
     [0.882353, 0.45098, 0.],
     [0.882353, 0.458824, 0.],
     [0.886275, 0.466667, 0.],
     [0.886275, 0.478431, 0.],
     [0.890196, 0.486275, 0.],
     [0.890196, 0.494118, 0.],
     [0.894118, 0.505882, 0.],
     [0.894118, 0.513725, 0.],
     [0.898039, 0.521569, 0.],
     [0.898039, 0.533333, 0.],
     [0.901961, 0.541176, 0.],
     [0.901961, 0.54902, 0.],
     [0.905882, 0.556863, 0.],
     [0.905882, 0.568627, 0.],
     [0.909804, 0.576471, 0.],
     [0.909804, 0.584314, 0.],
     [0.913725, 0.596078, 0.],
     [0.913725, 0.603922, 0.],
     [0.917647, 0.611765, 0.],
     [0.917647, 0.623529, 0.],
     [0.921569, 0.631373, 0.],
     [0.921569, 0.639216, 0.],
     [0.92549, 0.65098, 0.],
     [0.92549, 0.658824, 0.],
     [0.929412, 0.666667, 0.],
     [0.929412, 0.67451, 0.],
     [0.933333, 0.686275, 0.],
     [0.933333, 0.694118, 0.],
     [0.937255, 0.701961, 0.],
     [0.937255, 0.713725, 0.],
     [0.941176, 0.721569, 0.],
     [0.941176, 0.729412, 0.],
     [0.945098, 0.741176, 0.],
     [0.945098, 0.74902, 0.],
     [0.94902, 0.756863, 0.],
     [0.94902, 0.768627, 0.],
     [0.952941, 0.776471, 0.],
     [0.952941, 0.784314, 0.],
     [0.956863, 0.792157, 0.],
     [0.956863, 0.803922, 0.],
     [0.960784, 0.811765, 0.],
     [0.960784, 0.819608, 0.],
     [0.964706, 0.831373, 0.],
     [0.964706, 0.839216, 0.],
     [0.968627, 0.847059, 0.],
     [0.968627, 0.858824, 0.],
     [0.972549, 0.866667, 0.],
     [0.972549, 0.87451, 0.],
     [0.976471, 0.886275, 0.],
     [0.976471, 0.894118, 0.],
     [0.980392, 0.901961, 0.],
     [0.980392, 0.909804, 0.],
     [0.984314, 0.921569, 0.],
     [0.984314, 0.929412, 0.],
     [0.988235, 0.937255, 0.],
     [0.988235, 0.94902, 0.],
     [0.992157, 0.956863, 0.],
     [0.992157, 0.964706, 0.],
     [0.996078, 0.976471, 0.],
     [0.996078, 0.984314, 0.],
     [1., 0.992157, 0.],
     [1., 1., 0.]]
)