from typing import Optional  # noqa F401
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cv_depot.core.image import Image  # noqa: F401

import ipywidgets as ipy
import IPython.display as ipython
import re

import cv_depot.api as cvd
# ------------------------------------------------------------------------------


class ImageViewer:
    def __init__(self, image, size=81, gamma=1, premultiply=False):
        # type: (Image, int, float, bool) -> None
        '''
        Constructs an ImageViewer widget, used for displaying Image instances.

        Args:
            image (Image): Image instance.
            size (int, optional): Image size in percentage. Default: 81.
            gamma (float, optional): Initial gamma value. Default: 1.
            premultiply (bool, optional): Premultiply image by last channel.
                Default: False.

        Raises:
            EnforceError: If image is not an Image instance.
        '''
        # image
        self._image = image
        self.size = size
        self.gamma = gamma
        self.premult = premultiply

        # layer
        self.layer = self._get_layer_options()[0]
        self.layer_selector = ipy.Dropdown(
            description='layer',
            label=self.layer,
            options=self._get_layer_options(),
        )
        self.layer_selector.observe(self._handle_layer_event, names='value')

        # channel
        self.channel = self._get_channel_options()[0]
        self.channel_selector = ipy.Dropdown(
            description='channel',
            label=self.channel,
            options=self._get_channel_options(),
        )
        self.channel_selector.observe(self._handle_channel_event, names='value')

        # size slider
        self.size_slider = ipy.IntSlider(
            value=self.size, min=0, max=100, step=1, description='size'
        )
        self.size_slider.observe(self._handle_resize_event, names='value')

        # gamma slider
        self.gamma_slider = ipy.FloatSlider(
            value=self.gamma, min=0, max=10, step=0.01, description='gamma'
        )
        self.gamma_slider.observe(self._handle_gamma_event, names='value')

        self.premult_checkbox = ipy.Checkbox(
            value=False, description='premultiply',
        )
        self.premult_checkbox.observe(self._handle_premult_event, names='value')

        # viewer
        self.viewer = ipy.Image(value=self._get_png(), width=f'{self.size}%')
        self.info = ipy.HTML(value=self._get_info())

        # widgets
        sidebar = ipy.VBox(
            [
                self.info,
                self.layer_selector,
                self.channel_selector,
                self.size_slider,
                self.gamma_slider,
                self.premult_checkbox,
            ],
            layout=ipy.Layout(flex_flow='column', min_width='310px')
        )
        self._widgets = [
            ipy.HBox(
                [sidebar, self.viewer],
                layout=ipy.Layout(flex_flow='row')
            )
        ]

    def show(self):
        # type: () -> None
        '''
        Call ipython.display with widgets.
        '''
        ipython.display(*self._widgets)

    def _get_layer_options(self):
        # type: () -> list[str]
        '''
        Get list of channel layers.

        Returns:
            list[str]: List of channel layers.
        '''
        return self._image.channel_layers

    def _get_channel_options(self):
        # type: () -> list
        '''
        Get list of channel options.

        Returns:
            list: List of channel options.
        '''
        chan = self._image[:, :, self.layer].channels
        return ['all'] + chan

    def _get_info(self):
        # type: () -> str
        '''
        Creates a HTML representation of image info.

        Returns:
            str: HTML.
        '''
        info = self._image.info
        desc = ''
        for key in ['width', 'height', 'num_channels', 'bit_depth']:
            val = info[key]
            key = re.sub(' ', '&nbsp;', f'{key:>14}')
            desc += f'<span style="color: #7EC4CF;">{key}: </span>'
            desc += f'<span>{val}</span><br>'
        elem = '<p style="font-family: monospace; font-size: 13px; '
        elem += f'background: #242424;">{desc}</p>'
        return elem

    def _get_png(self):
        # type: () -> Optional[bytes]
        '''
        Creates a PNG representation of image data.

        Returns:
            str: PNG.
        '''
        chan = self.channel
        if chan == 'all':
            chan = self.layer
        chans = self._image[:, :, chan].channels
        if not self.premult and len(chans) > 3:
            chans = chans[:3]
        return cvd.ops.filter.gamma(self._image[:, :, chans], self.gamma)._repr_png()

    def _handle_layer_event(self, event):
        # type: (dict) -> None
        '''
        Handles layer selector events.

        Args:
            event (dict): Event.
        '''
        if event['type'] == 'change':
            # layer
            self.layer = event['new']
            self.layer_selector.value = self.layer

            # channel
            options = self._get_channel_options()
            chan = options[0]
            self.channel = chan
            self.channel_selector.options = options
            self.channel_selector.label = chan
            self.channel_selector.value = chan

            # viewer
            self.viewer.value = self._get_png()

    def _handle_channel_event(self, event):
        # type: (dict) -> None
        '''
        Handles channel selector events.

        Args:
            event (dict): Event.
        '''
        if event['type'] == 'change':
            self.channel = event['new']
            self.channel_selector.value = self.channel
            self.viewer.value = self._get_png()

    def _handle_resize_event(self, event):
        # type: (dict) -> None
        '''
        Handles image resize events.

        Args:
            event (dict): Event.
        '''
        if event['type'] == 'change':
            self.size = event['new']
            self.size_slider.value = self.size
            self.viewer.width = f'{self.size}%'

    def _handle_gamma_event(self, event):
        # type: (dict) -> None
        '''
        Handles image resize events.

        Args:
            event (dict): Event.
        '''
        if event['type'] == 'change':
            self.gamma = event['new']
            self.gamma_slider.value = self.gamma
            self.viewer.value = self._get_png()

    def _handle_premult_event(self, event):
        # type: (dict) -> None
        '''
        Handles premultiply events.

        Args:
            event (dict): Event.
        '''
        if event['type'] == 'change':
            self.premult = event['new']
            self.premult_checkbox.value = self.premult
            self.viewer.value = self._get_png()
