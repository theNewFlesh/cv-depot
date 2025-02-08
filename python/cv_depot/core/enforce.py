from typing import Iterable  # noqa F401

from lunchbox.enforce import Enforce
# ------------------------------------------------------------------------------


def enforce_homogenous_type(iterable, name='Iterable'):
    # type: (Iterable, str) -> None
    '''
    Ensures that iterable only contains only one type of object.

    Args:
        items (iterable): Iterable.
        name (str, optional): First word in error message. Default: Iterable.

    Raises:
        EnforceError: If iterable contains more than one type of object.
    '''
    types = [x.__class__.__name__ for x in iterable]
    types = sorted(list(set(types)))
    msg = f'{name} may only contain one type of object. Found types: {types}.'
    Enforce(len(types), '==', 1, message=msg)
