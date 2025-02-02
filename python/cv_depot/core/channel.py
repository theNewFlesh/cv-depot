from collections import OrderedDict
import re

from lunchbox.enforce import Enforce
# ------------------------------------------------------------------------------


class ChannelMap(OrderedDict):
    '''
    A channel to channel mapping between multiple source images and one target
    image.
    '''
    def __repr__(self):
        # type () -> str
        '''
        Returns a table of with columns: order, target and source.
        '''
        max_i = max(len(self.items()) - 1, len('order'))
        max_k = max(max(map(len, self.keys())), len('target'))
        header = f'{{i:<{max_i}}} {{k:<{max_k}}}       {{v}}'
        header = header.format(i='order', k='target', v='source')
        output = [header, '-' * len(header)]

        pattern = f'{{i:<{max_i}}} {{k:<{max_k}}}  <--  {{v}}'
        for i, (k, v) in enumerate(self.items()):
            output.append(pattern.format(i=i, k=k, v=v))
        output = '\n'.join(output)
        return output

    def __setitem__(self, key, value):
        # type: (str, str) -> None
        '''
        Sets given key to given value.
        Ensures that given key and value are legal.

        Args:
            key (str): Key.
            value (str): Value.

        Raises:
            EnforceError: If key is not a string.
            EnforceError: If value is not a string.
            EnforceError: If value is not one of: b, w, black, white,
                [frame].[channel]
        '''
        Enforce(key, 'instance of', str)
        Enforce(value, 'instance of', str)
        result = re.match(r'^(b|w|black|white|\d+\..+)$', value.lower())
        msg = 'ChannelMap values must match: b, w, black, white, or '
        msg += fr'[frame].[channel]. Illegal value: {value}.'
        Enforce(result, '!=', None, message=msg)

        super().__setitem__(key, value)

    @property
    def source(self):
        # type: () -> list[str]
        '''
        list[str]: Ordered list of source channels.
        '''
        return list(self.values())

    @property
    def target(self):
        # type: () -> list[str]
        '''
        list[str]: Ordered list of target channels.
        '''
        return list(self.keys())
