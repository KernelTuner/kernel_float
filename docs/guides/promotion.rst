Type Promotion
==============

For operations that involve two input arguments (or more), ``kernel_float`` will first convert the inputs into a common type before applying the operation.
For example, when adding ``vec<int, N>`` to a ``vec<float, N>``, both arguments must first be converted into a ``vec<float, N>``.

This procedure is called "type promotion" and is implemented as follows.
First, all arguments are converted into a vector by calling ``into_vec``.
Next, all arguments must have length ``N`` or length ``1`` and vectors of length ``1`` are resized to become length ``N``.
Finally, the vector element types are promoted into a common type.

The rules for element type promotion in ``kernel_float`` are slightly different than in regular C++.
In short, for two element types ``T`` and ``U``, the promotion rules can be summarized as follows:

* If one of the types is ``bool``, the result is the other type.
* If one type is a floating-point type and the other is a signed or unsigned integer, the result is the floating-point type.
* If both types are floating-point types, the result is the largest of the two types. An exception here is combining ``half`` and ``bfloat16``, which results in ``float``.
* If both types are integer types of the same signedness, the result is the largest of the two types.
* Combining a signed integer and unsigned integer type is not allowed.

Overview
--------

The type promotion rules are shown in the table below.
The labels are as follows:

* ``b``: boolean
* ``iN``: signed integer of ``N`` bits (e.g., ``int``, ``long``)
* ``uN``: unsigned integer of ``N`` bits (e.g., ``unsigned int``, ``size_t``)
* ``fN``: floating-point type of ``N`` bits (e.g., ``float``, ``double``)
* ``bf16``: bfloat16 floating-point format.

.. csv-table:: Type Promotion Rules.
   :file: promotion_table.csv
