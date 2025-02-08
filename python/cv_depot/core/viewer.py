from typing import Optional  # noqa F401
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cv_depot.core.image import Image  # noqa: F401

import ipywidgets as ipy
import IPython.display as ipython
import re
# ------------------------------------------------------------------------------


class ImageViewer:
    def __init__(self, image):
        # type: (Image) -> None
        '''
        Constructs an ImageViewer widget, used for displaying Image instances.

        Args:
            image (Image): Image instance.

        Raises:
            EnforceError: If image is not an Image instance.
        '''
        # image
        self._image = image

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

        # viewer
        self.viewer = ipy.Image(value=self._get_png(), width='87%')
        self.info = ipy.HTML(value=self._get_info())
        space = ipy.HTML(value='<div style="width: 115px;"></div>')

        # widgets
        self._widgets = [
            ipy.HBox(
                [space, self.layer_selector, self.channel_selector],
                layout=ipy.Layout(flex_flow='row')
            ),
            ipy.HBox(
                [self.info, self.viewer],
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
        desc = self._image._repr()
        desc = re.sub('<', '&lt;', desc)
        desc = re.sub('>', '&gt;', desc)
        desc = re.sub('\n', '<br>', desc)
        desc = re.sub(' ', '&nbsp;', desc)
        desc = f'<p style="font-family: monospace; font-size: 13px;">{desc}</p>'
        return desc

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
        return self._image[:, :, chan]._repr_png()

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
