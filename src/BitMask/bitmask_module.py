"""The BitMask class with all methods for creating and modifying
fixed-length binary data."""

# Author: Adam Thieringer
# Date: 2025-06-08
# Version: 1.2.0
# Bit hacks can be found at
# https://graphics.stanford.edu/~seander/bithacks.html


from __future__ import annotations

import warnings
from collections.abc import Generator
from typing_extensions import List, Self, Tuple, TypeAlias, Union

__all__ = ["BitMask"]
_MAX_BITMASK_LENGTH = 1024

IndexFormats: TypeAlias = Union[Tuple[int], List[int]]


def _value_validation(value: int, upper_bound: int) -> None:
    """
    Performs validation of input that would modify `BitMask.value`.

    :param value: The input to be verified.
    :param upper_bound: The `BitMask`'s largest integer that it can store.
    :raises TypeError: If `value` is not an integer
    :raises ValueError: If `value` is not positive or is larger than
        `upper_bound`.
    """
    if not isinstance(value, int):
        raise TypeError(f"Expected type int, got type {type(value).__name__}")

    if value < 0:
        raise ValueError(f"Value {value} must be positive")

    if value >= upper_bound:
        raise ValueError(f"Value {value} must be less than {upper_bound}")


def _index_validation(index: int | IndexFormats,
                      max_len: int) -> None:
    """
    Performs validation of input that attempts to modify or access an index or
    indices.

    :param index: An integer or list/tuple of integers to be verified.
    :param max_len: The `BitMask`'s maximum bit length.
    :raises TypeError: If `index` is not an integer or list/tuple of integers.
        If an `index` tuple/list contains a non-integer element.
    :raises IndexError: If `index` or any `index` element is less than `0` or
        greater than or equal to `max_len`.
    """
    if (not isinstance(index, int) and
            not isinstance(index, tuple) and
            not isinstance(index, list)):
        raise TypeError(f"Expected type int, tuple, or list, got type "
                        f"{type(index).__name__}")

    if isinstance(index, int):
        if index >= max_len or index < 0:
            raise IndexError(f"Index {index} is out of range")
    else:
        if not index:
            return

        for item in index:
            if not isinstance(item, int):
                raise TypeError("Tuples or lists of indices must only "
                                "contain integers")

            if item >= max_len or item < 0:
                raise IndexError(f"An index in {index} is out of range")


def _sliceable_index_validation(index: int | slice,
                                max_len) -> None:
    """
    Performs validation of input that would modify or access an index or
    indices.

    :param index: The input to be verified.
    :param max_len: The `BitMask`'s maximum bit length.
    :raises TypeError: If `index` is not an integer or slice.
    :raises ValueError: If `index` is an empty slice.
    :raises IndexError: If `index` is out of range.
    """
    if isinstance(index, slice):
        start, stop, step = index.indices(max_len)
        new_length = range(start, stop, step)

        if not new_length:
            raise ValueError(f"Cannot create a BitMask object from an "
                             f"empty slice {index}")
    elif isinstance(index, int):
        # Handle negative indices
        if abs(index) > max_len or index == max_len:
            raise IndexError(f"Index {index} is out of range")
    else:
        raise TypeError(
            f"BitMask indices must be integers or slices, not "
            f"{type(index).__name__}")


def _string_value_validation(value: str, base: int) -> int:
    """
    Performs validation of input string that would modify `BitMask.value`.

    :param value: The input to be verified.
    :param base: The base of the number the `value` string represents.
    :raises TypeError: If `value` is not a string.
    :raises ValueError: If `value` is not a valid string for the given `base`.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"Expected type str, got type {type(value).__name__}")

    try:
        int_value = int(value, base)
    except ValueError:
        base_strings = {
            2: "binary",
            10: "decimal",
            16: "hexadecimal"
        }

        raise ValueError(
            f"Value {value} is not a valid {base_strings[base]} string")

    return int_value


def _bit_mask_length_validation(other: BitMask,
                                max_len: int,
                                operation: str) -> None:
    """
    Confirms two `BitMask` instances have the same `max_length`.

    :param other: The other `BitMask` to be verified.
    :param max_len: The calling `BitMask`'s maximum bit length.
    :param operation: A string that indicates which operation is being called.
        `'>'`, `'<'`, `'^'`, `'|'`, etc.
    :raises TypeError: If `other.max_length` does not equal `max_len`.
    """
    if max_len != other.max_length:
        raise TypeError(
            f"{operation} not supported between instances of 'BitMask' with "
            f"different lengths")


def _shift_validation(other: int, left: bool = False) -> None:
    """
    Performs validation of input that would bit shift `BitMask.value`.

    :param other: The amount to shift.
    :param left: optional
        Whether the operation is shifting to the left or right.
    :raises ValueError: If `other` is negative.
        If the operation is shifting left and `other` is larger than `1024` to
        prevent overflows and excessive memory usage.
    """
    if not isinstance(other, int):
        return

    if other < 0:
        raise ValueError(f"Shift count {other} must be positive")

    if left and other > _MAX_BITMASK_LENGTH:
        raise ValueError(f"Cannot left shift more than 1024 bits, got {other}")


# Areas for optimization
# Any method that uses iteration
#    365         1   25276564.0    3e+07     14.8      reverse_bit_order()
#    369         1    9637643.0    1e+07      5.6      get_set_bits()
#    370         1   10422088.0    1e+07      6.1      get_cleared_bits()
#    383         1   53763237.0    5e+07     31.5      reversed()
#    385         1   31467011.0    3e+07     18.4      iter()
class BitMask:
    """
    Creates and modifies a numeric representation of fixed-length binary data.

    Each `BitMask` is defined by a fixed `max_length` and stores its data as a
    positive integer value. It provides functionalities for accessing and
    modifying bits (flags) by index, bitwise operations, or slicing. All
    methods index bits from the Least Significant Bit (LSB) to the Most
    Significant Bit (MSB).

    Examples:

    >>> # Create an 8-bit BitMask initialized to 0b00001011 (decimal 11)

    >>> mask = BitMask(8, 11)  # Create a BitMask: 0 0 0 0 1 0 1 1

    >>> mask

    0 0 0 0 1 0 1 1

    >>> mask[3]  # Access a bit (0-indexed from LSB)

    1

    >>> mask[-1] = 1  # Set a bit

    >>> mask

    1 0 0 0 1 0 1 1

    >>> mask[1:4]  # Access a slice

    1 0 1

    >>> # Perform a bitwise NOT (inverts all bits, returns new BitMask)

    >>> inverted_mask = ~mask

    >>> inverted_mask

    0 1 1 1 0 1 0 0
    """

    # --- Class Dunder Attributes ---
    # BitMask objects are mutable and unhashable by design
    __hash__ = None  # type: ignore[assignment]

    # --- Initialization ---
    def __init__(self, length: int, value: int = 0, *,
                 enable_validation: bool = True):
        """
        Initialize a BitMask object with a specified length and an optional
        initial integer value.

        Examples:

        >>> bm = BitMask(8, 15)

        >>> bm

        0 0 0 0 1 1 1 1

        >>> bm = BitMask(4, 5)

        >>> bm

        0 1 0 1

        :param length: The maximum number of bits the BitMask can hold.
        :param value: optional
            An integer value to initialize the BitMask (defaults to 0).
        :raises TypeError: If `length` or `value` are not integers.
        :raises ValueError: If `length` is negative.
            If `value` is not within the valid range (`0` to `2**length - 1`).
        """
        self._validation_enabled = enable_validation
        self._upper_bound = 1 << length

        if self._validation_enabled:
            if not isinstance(length, int):
                raise TypeError(
                    f"Expected type int, got type {type(length).__name__}")

            if length <= 0:
                raise ValueError(f"Length {length} must be greater than 0")
            elif length > _MAX_BITMASK_LENGTH:
                raise ValueError(f"Length {length} must be less than 1024, "
                                 f"consider using 'bytearray' or the "
                                 f"'bitarray' package")

            _value_validation(value, self._upper_bound)

        self._max_length = length
        self._value = value

    # --- Properties ---
    @property
    def max_length(self) -> int:
        """The maximum number of bits the BitMask can hold."""
        return self._max_length

    @property
    def value(self) -> int:
        """The current integer value of the BitMask."""
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        """
        Sets the integer value of the BitMask.

        Examples:

        >>> bm = BitMask(4)  # 0 0 0 0

        >>> bm.value = 13

        >>> bm

        1 0 1 1

        :param new_value: The new value of the BitMask.
        :raises TypeError: If `new_value` is not an integer.
        :raises ValueError: If `new_value` is not within the valid range (`0`
            to `2**self.max_length - 1`).
        """
        if self._validation_enabled:
            _value_validation(new_value, self._upper_bound)

        self._value = new_value

    # --- Bit Modifiers ---
    def set_bits(self, index: int | IndexFormats) -> None:
        """
        Sets the bit at the specified index or indices to `1`.

        Examples:

        >>> bm = BitMask(4)  # 0 0 0 0

        >>> bm.set_bits(3)

        >>> bm

        1 0 0 0

        :param index: The position of the bit to set.
        :raises TypeError: If `index` is not an integer.
        :raises IndexError: If `index` is not within the valid range (`0` to
            `self.max_length - 1`).
        """
        if self._validation_enabled:
            _index_validation(index, self._max_length)

        if isinstance(index, int):
            self._value |= 1 << index
        else:
            for item in index:
                self._value |= 1 << item

    def set_all(self) -> None:
        """
        Set all bits in the BitMask to `1`.

        Examples:

        >>> bm = BitMask(4)  # 0 0 0 0

        >>> bm.set_all()

        >>> bm

        1 1 1 1
        """
        self._value |= self._upper_bound - 1

    def flip_bits(self, index: int | IndexFormats) -> None:
        """
        Flips the bit at the specified index or indices.

        Examples:

        >>> bm = BitMask(4, 5)  # 0 1 0 1

        >>> bm.flip_bits(0)

        >>> bm

        0 1 0 0

        >>> bm.flip_bits(3)

        >>> bm

        1 1 0 0

        :param index: The position of the bit to flip.
        :raises TypeError: If `index` is not an integer.
        :raises IndexError: If `index` is not within the valid range (`0` to
            `self.max_length - 1`).
        """
        if self._validation_enabled:
            _index_validation(index, self._max_length)

        if isinstance(index, int):
            self._value ^= 1 << index
        else:
            for item in index:
                self._value ^= 1 << item

    def flip_all(self) -> None:
        """
        Flips all bits in the BitMask.

        Examples:

        >>> bm = BitMask(4, 9)  # 1 0 0 1

        >>> bm.flip_all()

        >>> bm

        0 1 1 0
        """
        self._value ^= self._upper_bound - 1

    def clear_bits(self, index: int | IndexFormats) -> None:
        """
        Sets the bit at the specified index or indices to `0`.

        Examples:

        >>> bm = BitMask(4, 15)  # 1 1 1 1

        >>> bm.clear_bits(2)

        >>> bm

        1 0 1 1

        :param index: The position of the bit to clear.
        :raises TypeError: If `index` is not an integer.
        :raises IndexError: If `index` is not within the valid range (`0` to
            `self.max_length - 1`).
        """
        if self._validation_enabled:
            _index_validation(index, self._max_length)

        if isinstance(index, int):
            self._value &= ~(1 << index)
        else:
            for item in index:
                self._value &= ~(1 << item)

    def clear_all(self) -> None:
        """
        Sets all bits in the BitMask to `0`.

        Examples:

        >>> bm = BitMask(4, 15)  # 1 1 1 1

        >>> bm.clear_all()

        >>> bm

        0 0 0 0
        """
        self._value = 0

    def reverse_bit_order(self) -> None:
        """
        Reverses the bit order of the BitMask.

        Examples:

        >>> bm = BitMask(8, 168)  # 1 0 1 0 1 0 0 0

        >>> bm.reverse_bit_order()

        >>> bm

        0 0 0 1 0 1 0 1
        """
        reversed_bits = 0
        for i in range(self._max_length):
            reversed_bits <<= 1
            reversed_bits |= ((self._value >> i) & 1)

        self._value = reversed_bits

    # --- Information Getters ---
    def get_bits(self, index: int | IndexFormats) -> int | tuple[int, ...]:
        """
        Returns the value of the bit(s) at the specified index or indices.

        Examples:

        >>> bm = BitMask(4, 7)  # 0 1 1 1

        >>> bm.get_bits(0)

        1

        >>> bm.get_bits((1, 3))

        (1, 0)

        :param index: The index or list of indices of the bit(s) to retrieve.
        :return: The value of the bit(s) (`0` or `1`).
        :raises TypeError: If `index` is not an integer.
        :raises IndexError: If `index` is not within the valid range (`0` to
            `self.max_length - 1`).
        """
        if self._validation_enabled:
            _index_validation(index, self._max_length)

        if isinstance(index, int):
            return (self._value >> index) & 1
        else:
            return tuple((self._value >> item) & 1 for item in index)

    def get_count(self) -> int:
        """
        Returns the number of set bits (`1`s) in the BitMask.

        Examples:

        >>> bm = BitMask(8, 210)  # 1 1 0 1 0 0 1 0

        >>> bm.get_count()

        4

        :return: The number of set bits.
        """
        temp_value = self._value
        count = 0
        while temp_value:
            temp_value &= temp_value - 1
            count += 1

        return count

    def get_set_bits(self) -> tuple[int, ...]:
        """
        Returns the indices of all set bits (`1`s).

        Examples:

        >>> bm = BitMask(8, 210)  # 1 1 0 1 0 0 1 0

        >>> bm.get_set_bits()

        (1, 4, 6, 7)

        >>> bm = BitMask(8)  # 0 0 0 0 0 0 0 0

        >>> bm.get_set_bits()

        ()

        :return: A tuple of indices where bits are equal to `1`.
        """
        return tuple(index for index in range(self._max_length) if
                     (self._value >> index) & 1)

    def get_cleared_bits(self) -> tuple[int, ...]:
        """
        Returns the indices of all cleared bits (`0`s).

        Examples:

        >>> bm = BitMask(8, 210)  # 1 1 0 1 0 0 1 0

        >>> bm.get_cleared_bits()

        (0, 2, 3, 5)

        >>> bm = BitMask(8, 255)  # 1 1 1 1 1 1 1 1

        >>> bm.get_cleared_bits()

        ()

        :return: A tuple of indices where bits are equal to `0`.
        """
        return tuple(index for index in range(self._max_length) if
                     not (self._value >> index) & 1)

    def get_lsb(self) -> int:
        """
        Returns the index of the least significant set bit (LSB).

        Examples:

        >>> bm = BitMask(8, 210)  # 1 1 0 1 0 0 1 0

        >>> bm.get_lsb()

        1

        >>> bm = BitMask(8)  # 0 0 0 0 0 0 0 0

        >>> bm.get_lsb()

        -1

        :return: The index of the LSB, or `-1` if all bits are `0`.
        """
        return (self._value & -self._value).bit_length() - 1

    def get_msb(self) -> int:
        """
        Returns the index of the most significant set bit (MSB).

        Examples:

        >>> bm = BitMask(8, 210)  # 1 1 0 1 0 0 1 0

        >>> bm.get_msb()

        7

        >>> bm = BitMask(8)  # 0 0 0 0 0 0 0 0

        >>> bm.get_msb()

        -1

        :return: The index of the MSB, or `-1` if all bits are `0`.
        """
        return self._value.bit_length() - 1

    # --- self.value String Setters ---
    def set_binary(self, new_value: str) -> None:
        """
        Sets `value` using a binary string.

        Examples:

        >>> bm = BitMask(4)  # 0 0 0 0

        >>> bm.set_binary("0b1100")

        >>> bm

        1 1 0 0

        >>> bm.set_binary("1010")

        >>> bm

        1 0 1 0

        :param new_value: A binary string representation of the new value.
        :raises TypeError: If `new_value` is not a string.
        :raises ValueError: If `new_value` is not within the valid range (`0`
            to `2**self.max_length - 1`).
            If `new_value` cannot be interpreted as a binary string.
        """
        if self._validation_enabled:
            int_value = _string_value_validation(new_value, 2)

            _value_validation(int_value, self._upper_bound)

            self._value = int_value
        else:
            self._value = int(new_value, 2)

    def set_hexadecimal(self, new_value: str) -> None:
        """
        Sets `value` using a hexadecimal string.

        Examples:

        >>> bm = BitMask(5)  # 0 0 0 0 0

        >>> bm.set_hexadecimal("0x1A")

        >>> bm

        1 1 0 1 0

        >>> bm.set_hexadecimal("0D")

        >>> bm

        0 1 1 0 1

        :param new_value: A hexadecimal string representation of the value.
        :raises TypeError: If `new_value` is not a string.
        :raises ValueError: If `new_value` is not within the valid range (`0`
            to `2**self.max_length - 1`).
            If `new_value` cannot be interpreted as a hexadecimal string.
        """
        if self._validation_enabled:
            int_value = _string_value_validation(new_value, 16)

            _value_validation(int_value, self._upper_bound)

            self._value = int_value
        else:
            self._value = int(new_value, 16)

    def set_decimal(self, new_value: str) -> None:
        """
        Sets `value` using a decimal integer string.

        Examples:

        >>> bm = BitMask(4)  # 0 0 0 0

        >>> bm.set_decimal("6")

        >>> bm

        0 1 1 0

        >>> bm.set_decimal("03")

        >>> bm

        0 0 1 1

        :param new_value: A string representation of a decimal integer to set
            `BitMask.value` to.
        :raises TypeError: If `new_value` is not a string.
        :raises ValueError: If `new_value` is not within the valid range (`0`
            to `2**self.max_length - 1`).
            If `new_value` cannot be interpreted as a decimal string.
        """
        if self._validation_enabled:
            int_value = _string_value_validation(new_value, 10)

            _value_validation(int_value, self._upper_bound)

            self._value = int_value
        else:
            self._value = int(new_value)

    # --- self.value String Formats ---
    def to_binary(self) -> str:
        """Returns a binary string representation of `self.value`."""
        return "0b" + bin(self._value)[2:].zfill(self._max_length)

    def to_hexadecimal(self) -> str:
        """Returns a hexadecimal string representation of `self.value`."""
        hexadecimal_string = hex(self._value)
        return hexadecimal_string[:2] + hexadecimal_string[2:].zfill(
            (self._max_length + 3) >> 2)

    def to_decimal(self) -> str:
        """Returns a decimal string representation of `self.value`."""
        return str(self._value)

    # --- Representations ---
    def __repr__(self) -> str:
        return (f"BitMask({self._max_length}, {self._value}, enable_validation"
                f"={self._validation_enabled})")

    def __str__(self) -> str:
        """Returns a space-separated string of bits, from MSB to LSB"""
        return ' '.join(bin(self._value)[2:].zfill(self._max_length))

    # --- Iterator Methods ---
    def __len__(self) -> int:
        """Returns `self.max_length`"""
        return self._max_length

    def __getitem__(self, item: int | slice) -> int | BitMask:
        """
        Returns the value of a bit at a specific index or a new `BitMask`
        object from a slice.

        When `item` is an integer, returns the value (`0` or `1`) of the bit at
        that index. Supports negative indexing.

        When `item` is a slice, returns a new `BitMask` object representing the
        bits within the specified slice. The new `BitMask`'s LSB corresponds to
        the first bit of the slice.

        Examples:

        >>> bm = BitMask(8, 140)  # 1 0 0 0 1 1 0 0

        >>> bm[5]

        0

        >>> bm[2:6]

        0 0 1 1

        >>> bm[::-2]

        0 1 0 1

        :param item: An integer index or a slice object.
        :return: An integer (0 or 1) if `item` is an index, or a new `BitMask`
            object if `item` is a slice.
        :raises TypeError: If `item` is not an integer or slice.
        :raises IndexError: If an integer `item` index is out of bounds.
        :raises ValueError: If `item` represents an empty slice.
        """
        if self._validation_enabled:
            _sliceable_index_validation(item, self._max_length)

        if isinstance(item, slice):
            start, stop, step = item.indices(self._max_length)
            new_value = 0
            new_length = len(range(start, stop, step))

            for index, pos in enumerate(range(start, stop, step)):
                new_value |= ((self._value >> pos) & 1) << index

            return BitMask(new_length, new_value,
                           enable_validation=self._validation_enabled)
        else:
            # Handle negative indices
            if item < 0:
                item += self._max_length

            return (self._value >> item) & 1

    def __setitem__(self, key: int | slice, value: int) -> None:
        """
        Modifies the bit(s) at the specified index or slice.

        When `key` is an integer, `value` must be `0` or `1`.
        When `key` is a slice, the bits of `value` are applied to the slice.
        The least significant bit (LSB) of `value` corresponds to the first bit
        in the slice, moving towards the most significant bit (MSB) of `value`.

        A :py:class:`RuntimeWarning` is issued if the `value` integer has more
        set bits than the length of the target slice, indicating that some of
        `value`'s most significant bits will be truncated.

        Examples:

        >>> bm = BitMask(8)  # 0 0 0 0 0 0 0 0

        >>> bm[5] = 1

        >>> bm

        0 0 0 1 0 0 0 0

        >>> bm[2:6] = 11  # 1011

        >>> bm

        0 0 1 0 1 1 0 0

        >>> bm[::-2] = 9  # 1001

        >>> bm

        1 0 0 0 0 1 1 0

        :param key: An integer index or a slice object specifying the bit(s) to
            modify.
        :param value: The value to assign to the specified bit(s), as an
            integer.
        :raises TypeError: If `key` is not an integer or slice.
        :raises IndexError: If an integer `key` (or its absolute value, for
            negative indices) is out of bounds.
        :raises ValueError: If `value` is negative (when `key` is a slice), or
            if `value` is not `0` or `1` (when `key` is an integer).
            If `key` represents an empty slice.
        """
        if self._validation_enabled:
            _sliceable_index_validation(key, self._max_length)

        if value < 0:
            raise ValueError(f"Value {value} must be positive")

        if isinstance(key, slice):
            start, stop, step = key.indices(self._max_length)
            slice_length = len(range(start, stop, step))

            if value.bit_length() > slice_length:
                warnings.warn(
                    f"Value {value} has more bits "
                    f"'{value.bit_length()}' than slice '{slice_length}'",
                    RuntimeWarning, stacklevel=2)

            for index, bit in enumerate(range(start, stop, step)):
                if (value >> index) & 1:
                    self._value |= 1 << bit
                else:
                    self._value &= ~(1 << bit)
        elif isinstance(key, int):
            # Handle negative indices
            if key < 0:
                key += self._max_length

            if self._validation_enabled:
                if value not in [0, 1]:
                    raise ValueError(
                        f"Cannot assign {value} to position {key} because "
                        f"{value} is not '0' or '1'")

            if value:
                self._value |= 1 << key
            else:
                self._value &= ~(1 << key)

    def __iter__(self) -> Generator[int]:
        """
        Yields each bit value from the Least Significant Bit (LSB) to Most
        Significant Bit (MSB).

        Examples:

        >>> bm = BitMask(6, 26)  # 0 1 1 0 1 0

        >>> list(bm)

        [0, 1, 0, 1, 1, 0]

        :return: A generator object that yields integer bit values
            (`0` or `1`).
        """
        current = 0
        while current < self._max_length:
            yield (self._value >> current) & 1
            current += 1

    def __reversed__(self) -> Generator[int]:
        """
        Yields each bit value from the Most Significant Bit (MSB) to Least
        Significant Bit (LSB).

        Examples:

        >>> bm = BitMask(6, 26)  # 0 1 1 0 1 0

        >>> list(reversed(bm))

        [0, 1, 1, 0, 1, 0]
        """

        current = self._max_length - 1
        while current >= 0:
            yield (self._value >> current) & 1
            current -= 1

    # --- Comparisons ---
    def __eq__(self, other: object) -> bool:
        """
        Compares the `BitMask` object for equality with another object.

        When `other` is an integer, returns `True` if `self.value` is equal to
        `other`. When `other` is a `BitMask` object, returns `True` if both
        `self.value` is equal to `other.value` and `self.max_length` is equal
        to `other.max_length`.

        Examples:

        >>> bm = BitMask(8, 255)  # 1 1 1 1 1 1 1 1

        >>> bm == 255

        True

        >>> bm == BitMask(12, 255)  # 0 0 0 0 1 1 1 1 1 1 1 1

        False

        >>> bm == BitMask(8, 15)  # 0 0 0 0 1 1 1 1

        False

        :param other: The object to compare against. Can be an integer or
            `BitMask` object.
        :return: `True` if the requirements for equality are satisfied, `False`
            otherwise. Returns `NotImplemented` if `other` is not an integer or
            `BitMask` object.
        """
        if isinstance(other, int):
            return self._value == other
        elif isinstance(other, BitMask):
            return (self._value == other.value and
                    self._max_length == other._max_length)
        else:
            return NotImplemented

    def __ne__(self, other: object) -> bool:
        """
        Compares the `BitMask` object for inequality with another object.

        Returns the inverse of `__eq__()`

        :param other: The object to compare against. Can be an integer or
            `BitMask` object.
        :return: `True` if the requirements for equality are not satisfied,
            `False` otherwise. Returns `NotImplemented` if `other` is not an
            integer or `BitMask` object.
        """
        equality = self.__eq__(other)
        if equality == NotImplemented:
            return NotImplemented
        else:
            return not equality

    def __lt__(self, other: int | BitMask) -> bool:
        """
        Compares the `BitMask` object's `self.value` for strict inequality
        (less than) with another object.

        When `other` is an integer, returns `True` if `self.value` is less than
        `other`. When `other` is a `BitMask` object with the same `max_length`,
        returns `True` if `self.value` is less than `other.value`.

        Examples:

        >>> bm = BitMask(4, 5)  # 0 1 0 1

        >>> bm < 5

        False

        >>> bm < BitMask(4, 7)  # 0 1 1 1

        True

        :param other: An integer or `BitMask` to be compared.
        :return: `True` if the requirements for strict inequality are
            satisfied, `False` otherwise. Returns `NotImplemented` if `other`
            is not an integer or `BitMask` object.
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        """
        if isinstance(other, int):
            return self._value < other
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "'<'")

            return self._value < other.value
        else:
            return NotImplemented

    def __le__(self, other: int | BitMask) -> bool:
        """
        Compares the `BitMask` object's `self.value` for strict inequality
        (less than or equal to) with another object.

        When `other` is an integer, returns `True` if `self.value` is less than
        or equal to `other`. When `other` is a `BitMask` object with the same
        `max_length`, returns `True` if `self.value` is less than or equal to
        `other.value`.

        Examples:

        >>> bm = BitMask(4, 5)  # 0 1 0 1

        >>> bm <= 5

        True

        >>> bm <= BitMask(4, 3)  # 0 0 1 1

        False

        :param other: An integer or `BitMask` to be compared.
        :return: `True` if the requirements for strict inequality are
            satisfied, `False` otherwise. Returns `NotImplemented` if `other`
            is not an integer or `BitMask` object.
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        """
        if isinstance(other, int):
            return self._value <= other
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "'<='")

            return self._value <= other.value
        else:
            return NotImplemented

    def __gt__(self, other: int | BitMask) -> bool:
        """
        Compares the `BitMask` object's `self.value` for strict inequality
        (greater than) with another object.

        When `other` is an integer, returns `True` if `self.value` is greater
        than `other`. When `other` is a `BitMask` object with the same
        `max_length`, returns `True` if `self.value` is greater than
        `other.value`.

        Examples:

        >>> bm = BitMask(4, 5)  # 0 1 0 1

        >>> bm > 2

        True

        >>> bm > BitMask(4, 7)  # 0 1 1 1

        False

        :param other: An integer or `BitMask` to be compared.
        :return: `True` if the requirements for strict inequality are
            satisfied, `False` otherwise. Returns `NotImplemented` if `other`
            is not an integer or `BitMask` object.
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        """
        if isinstance(other, int):
            return self._value > other
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "'>'")

            return self._value > other.value
        else:
            return NotImplemented

    def __ge__(self, other: int | BitMask) -> bool:
        """
        Compares the `BitMask` object's `self.value` for strict inequality
        (greater than or equal to) with another object.

        When `other` is an integer, returns `True` if `self.value` is greater
        than or equal to `other`. When `other` is a `BitMask` object with the
        same `max_length`, returns `True` if `self.value` is greater than or
        equal to `other.value`.

        Examples:

        >>> bm = BitMask(4, 5)  # 0 1 0 1

        >>> bm >= 5

        True

        >>> bm >= BitMask(4, 4)  # 0 1 0 0

        False

        :param other: An integer or `BitMask` to be compared.
        :return: `True` if the requirements for strict inequality are
            satisfied, `False` otherwise. Returns `NotImplemented` if `other`
            is not an integer or `BitMask` object.
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        """
        if isinstance(other, int):
            return self._value >= other
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "'>='")

            return self._value >= other.value
        else:
            return NotImplemented

    # --- Unary Operations ---
    def __invert__(self) -> BitMask:
        """
        Returns a `BitMask` object with max length, `self.max_length`, where
        all bits are flipped.

        Examples:

        >>> bm = BitMask(8, 105)  # 0 1 1 0 1 0 0 1

        >>> ~bm

        1 0 0 1 0 1 1 0

        """
        return BitMask(self._max_length,
                       self._value ^ (self._upper_bound - 1),
                       enable_validation=self._validation_enabled)

    # --- Bitwise Operations ---
    def __and__(self, other: int | BitMask) -> BitMask:
        """
        Performs a bitwise AND operation (`&`) with another object, returning a
        new `BitMask`.

        When `other` is an integer, the operation is performed directly.
        When `other` is another `BitMask` object, the operation is performed
        with `other.value`.

        Returns a new `BitMask` object with `self.max_length` and the resulting
        value.

        Examples:

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm & 2  # 0 0 1 0

        0 0 1 0

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm2 = BitMask(4, 13)  # 1 1 0 1

        >>> bm & bm2

        0 1 0 0

        :param other: An integer or `BitMask` object to perform the bitwise AND
            operation with.
        :return: A new `BitMask` object representing the result of the bitwise
            AND.
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        :raises ValueError: If `other` is an integer and is not within the
            valid range (`0` to `(2**self.max_length) - 1`).
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _value_validation(other, self._upper_bound)

            return BitMask(self._max_length, self._value & other,
                           enable_validation=self._validation_enabled)
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "Bitwise operations")

            return BitMask(self._max_length, self._value & other.value,
                           enable_validation=self._validation_enabled)
        else:
            return NotImplemented

    def __or__(self, other: int | BitMask) -> BitMask:
        """
        Performs a bitwise OR operation (`|`) with another object, returning a
        new `BitMask`.

        When `other` is an integer, the operation is performed directly.
        When `other` is another `BitMask` object, the operation is performed
        with `other.value`.

        Returns a new `BitMask` object with `self.max_length` and the resulting
        value.

        Examples:

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm | 10  # 1 0 1 0

        1 1 1 0

        >>> bm = BitMask(4, 2)  # 0 0 1 0

        >>> bm2 = BitMask(4, 10)  # 1 0 0 1

        >>> bm | bm2

        1 0 1 1

        :param other: An integer or `BitMask` object to perform the bitwise OR
            operation with.
        :return: A new `BitMask` object representing the result of the bitwise
            OR.
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        :raises ValueError: If `other` is an integer and is not within the
            valid range (`0` to `(2**self.max_length) - 1`).
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _value_validation(other, self._upper_bound)

            return BitMask(self._max_length, self._value | other,
                           enable_validation=self._validation_enabled)
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "Bitwise operations")

            return BitMask(self._max_length, self._value | other.value,
                           enable_validation=self._validation_enabled)
        else:
            return NotImplemented

    def __xor__(self, other: int | BitMask) -> BitMask:
        """
        Performs a bitwise XOR operation (`^`) with another object, returning a
        new `BitMask`.

        When `other` is an integer, the operation is performed directly.
        When `other` is another `BitMask` object, the operation is performed
        with `other.value`.

        Returns a new `BitMask` object with `self.max_length` and the resulting
        value.

        Examples:

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm ^ 10  # 1 0 1 0

        1 1 0 0

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm2 = BitMask(4, 13)  # 1 1 0 1

        >>> bm ^ bm2

        1 0 1 1

        :param other: An integer or `BitMask` object to perform the bitwise XOR
            operation with.
        :return: A new `BitMask` object representing the result of the bitwise
            XOR.
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        :raises ValueError: If `other` is an integer and is not within the
            valid range (`0` to `(2**self.max_length) - 1`).
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _value_validation(other, self._upper_bound)

            return BitMask(self._max_length, self._value ^ other,
                           enable_validation=self._validation_enabled)
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "Bitwise operations")

            return BitMask(self._max_length, self._value ^ other.value,
                           enable_validation=self._validation_enabled)
        else:
            return NotImplemented

    def __lshift__(self, other: int) -> BitMask:
        """
        Returns a new `BitMap` with `length`, `self.max_length` and `value`,
        `self.value << other`.

        :param other: The number of places to shift.
        :return: A new `BitMask` object with a left shifted `value`.
        :raises ValueError: If `other` is not positive.
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _shift_validation(other, True)

            # Removes all bits past the BitMask's upper limit
            return BitMask(self._max_length, (self._value << other) & (
                    self._upper_bound - 1),
                           enable_validation=self._validation_enabled)
        else:
            return NotImplemented

    def __rshift__(self, other: int) -> BitMask:
        """
        Returns a new `BitMap` with `length`, `self.max_length` and `value`,
        `self.value >> other`.

        :param other: The number of places to shift.
        :return: A new `BitMask` object with a right shifted `value`.
        :raises ValueError: If `other` is not positive.
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _shift_validation(other)

            return BitMask(self._max_length, self._value >> other,
                           enable_validation=self._validation_enabled)
        else:
            return NotImplemented

    # --- In-place Bitwise Operations ---
    def __iand__(self, other: int | BitMask) -> Self:
        """
        Performs a bitwise AND operation (`&`) with another object, and sets
        `self.value` to the result.

        When `other` is an integer, the operation is performed directly.
        When `other` is another `BitMask` object, the operation is performed
        with `other.value`.

        Returns the current `BitMask` object, with the changed `value`.

        Examples:

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm &= 2  # 0 0 1 0

        >>> bm

        0 0 1 0

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm2 = BitMask(4, 13)  # 1 1 0 1

        >>> bm &= bm2

        >>> bm

        0 1 0 0

        :param other: An integer or `BitMask` object to perform the bitwise AND
            operation with.
        :return: `self`
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        :raises ValueError: If `other` is an integer and is not within the
            valid range (`0` to `(2**self.max_length) - 1`).
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _value_validation(other, self._upper_bound)

            self._value &= other
            return self
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "Bitwise operations")

            self._value &= other.value
            return self
        else:
            return NotImplemented

    def __ior__(self, other: int | BitMask) -> Self:
        """
        Performs a bitwise OR operation (`|`) with another object, and sets
        `self.value` to the result.

        When `other` is an integer, the operation is performed directly.
        When `other` is another `BitMask` object, the operation is performed
        with `other.value`.

        Returns the current `BitMask` object, with the changed `value`.

        Examples:

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm |= 10  # 1 0 1 0

        >>> bm

        1 1 1 0

        >>> bm = BitMask(4, 2)  # 0 0 1 0

        >>> bm2 = BitMask(4, 10)  # 1 0 0 1

        >>> bm |= bm2

        >>> bm

        1 0 1 1

        :param other: An integer or `BitMask` object to perform the bitwise OR
            operation with.
        :return: `self`
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        :raises ValueError: If `other` is an integer and is not within the
            valid range (`0` to `(2**self.max_length) - 1`).
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _value_validation(other, self._upper_bound)

            self._value |= other
            return self
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "Bitwise operations")

            self._value |= other.value
            return self
        else:
            return NotImplemented

    def __ixor__(self, other: int | BitMask) -> Self:
        """
        Performs a bitwise XOR operation (`^`) with another object, and sets
        `self.value` to the result.

        When `other` is an integer, the operation is performed directly.
        When `other` is another `BitMask` object, the operation is performed
        with `other.value`.

        Returns the current `BitMask` object, with the changed `value`.

        Examples:

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm ^= 2  # 0 0 1 0

        >>> bm

        0 1 0 0

        >>> bm = BitMask(4, 6)  # 0 1 1 0

        >>> bm2 = BitMask(4, 13)  # 1 1 0 1

        >>> bm ^= bm2

        >>> bm

        1 0 1 1

        :param other: An integer or `BitMask` object to perform the bitwise XOR
            operation with.
        :return: `self`
        :raises TypeError: If `other` is a `BitMask` and its `max_length` is
            not equal to `self.max_length`.
        :raises ValueError: If `other` is an integer and is not within the
            valid range (`0` to `(2**self.max_length) - 1`).
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _value_validation(other, self._upper_bound)

            self._value ^= other
            return self
        elif isinstance(other, BitMask):
            if self._validation_enabled:
                _bit_mask_length_validation(other, self._max_length,
                                            "Bitwise operations")

            self._value ^= other.value
            return self
        else:
            return NotImplemented

    def __ilshift__(self, other: int) -> Self:
        """
        Shifts `self.value` `other` bits to the left.

        :param other: The number of places to shift.
        :return: `self`.
        :raises ValueError: If `other` is not positive.
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _shift_validation(other, True)

            # Removes all bits past the BitMask's upper limit
            self._value = (self._value << other) & (self._upper_bound - 1)
            return self
        else:
            return NotImplemented

    def __irshift__(self, other: int) -> Self:
        """
        Shifts `self.value` `other` bits to the right.

        :param other: The number of places to shift.
        :return: `self`.
        :raises ValueError: If `other` is not positive.
        """
        if isinstance(other, int):
            if self._validation_enabled:
                _shift_validation(other)

            self._value >>= other
            return self
        else:
            return NotImplemented

    # --- Casting ---
    def __bool__(self):
        """Returns `True` if any bit is set, otherwise `False`"""
        return True if self._value else False

    def __int__(self):
        """Returns `self.value`"""
        return self._value

    def __float__(self):
        """Returns the float representation of `self.value`"""
        return float(self._value)
