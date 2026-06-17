"""Generic functions for validity checks"""

def _check_(test, **kwargs):
    """Apply generic assert test to all input keywords arguments excepting to `None` occurences.

    Parameters
    ----------

    test : <class 'function'>
        generic test that will be applied to all non ``None`` values
        in `kwargs`, return type must be ``bool``. 

    **kwargs : keywords arguments
        Input keyword arguments to be tested. 

        Each keyword value must be either `None` or an array_like
        (this will not be tested).


    Return
    ------

    ko : tuple
        A tuple containing the elements ``key in kwargs.keys()`` such
        that ``kwargs['key'] is not None and test(kwargs['key']) is
        False``.

    Note
    ----

    The compatibility between the non ``None`` elements of
    ``kwargs.values()`` and the provided `test` function is **not
    verified** by this function.

    """
    #out = {key: test(value) for key, value in args.items() if value is not None}
    #ok = tuple(key for key, value in out.items() if value)
    #ko = tuple(key for key, value in out.items() if not value)
    ko = tuple(key for key, value in kwargs.items() if value is not None and not test(value))
    return ko

def _check_same_dtype_(**kwargs):
    """Check whether or not all non `None` keyword parameters have same array data type.

    Parameters
    ----------

    **kwargs : keyword arguments
        Input keyword arguments to be tested. 

        Each keyword value must be either `None` or an array_like
        (this will not be tested).


    Return
    ------

    no_error_flag : bool
        True if the test is successful (otherwise an exception is
        raised).

    """
    args2 = {key: value for key, value in kwargs.items() if value is not None}
    if len(args2) > 1:
        dtype = list(args2.values())[0].dtype
        if not all([a.dtype == dtype for a in args2.values()]):
            raise RuntimeError(
                "Parameters %s must have the same data type." % str(list(args2.keys()))
            )        
    return True
    
def _check_type_(t, **kwargs):
    """Check whether or not all non `None` keyword parameters are instances of the specified type.

    Parameters
    ----------

    t : <class 'type'>
        The specified type

    **kwargs : keyword arguments
        Input keyword arguments to be tested. 
    
    
    Return
    ------

    no_error_flag : bool
        True if the test is successful (otherwise an exception is
        raised).

    """
    ko = _check_(lambda x : isinstance(x, t), **kwargs)
    if len(ko) > 1:
        raise RuntimeError("Inconsistent type for parameters %s (expected `%s`)." % (str(ko), t))
    elif len(ko) == 1:
        raise RuntimeError("Inconsistent type for parameter `%s` (expected `%s`)." % (ko[0], t))
    return True

def _check_dtype_(dtype, **kwargs):
    """Check whether or not all non `None` keyword parameters have the specified array data type.

    Parameters
    ----------

    dtype : <class 'type'> or <class 'torch.type'>
        The specified array data type

    **kwargs : keyword arguments
        Input keyword arguments to be tested. 

        Each keyword value must be either `None` or an array_like
        (this will not be tested).


    Return
    ------

    no_error_flag : bool
        True if the test is successful (otherwise an exception is
        raised).

    """
    ko = _check_(lambda x : dtype == x.dtype, **kwargs)
    if len(ko) > 1:
        raise RuntimeError("Inconsistent data type for parameters %s (expected `%s`)." % (str(ko), dtype))
    elif len(ko) == 1:
        raise RuntimeError("Inconsistent data type for parameter `%s` (expected `%s`)." % (ko[0], dtype))
    return True

def _check_ndim_(ndim, **kwargs):
    """Check whether or not all non `None` keyword parameters have the specified number of array dimensions.

    Parameters
    ----------

    ndim : int
        The specified number of array dimensions.

    **kwargs : keyword arguments
        Input keyword arguments to be tested. 

        Each keyword value must be either `None` or an array_like
        (this will not be tested).

    Return
    ------

    no_error_flag : bool
        True if the test is successful (otherwise an exception is
        raised).

    """
    ko = _check_(lambda x : ndim == x.ndim, **kwargs)
    if len(ko) > 1:
        raise RuntimeError(
            "Inconsistent dimensions for parameters %s (expected `ndim=%d`)." % (str(ko), ndim)
        )
    elif len(ko) == 1: 
        raise RuntimeError(
            "Inconsistent dimensions for parameter `%s` (expected `ndim=%d`)." % (ko[0], ndim)
        )
    return True

def _check_backend_(backend, **kwargs):
    """Check whether or not all non `None` keyword parameters array belongs to the same library (torch, numpy, cupy).

    Parameters
    ----------

    backend : <class 'pyepri.backends.Backend'>
        A `numpy`, `cupy` or `torch` backend instance.

    **kwargs : keyword arguments
        Input keyword arguments to be tested. 

        Each keyword value must be either `None` or an array_like
        (this will not be tested).

    Return
    ------

    no_error_flag : bool
        True if the test is successful (otherwise an exception is
        raised).

    """
    ko = _check_(lambda x : isinstance(x, backend.cls), **kwargs)
    if len(ko) > 1:
        raise RuntimeError(
            "Parameters %s are not consistent with the provided backend. Since\n"
            "`backend.lib` is `%s`, those parameters must all be %s instances.\n"
            % (str(ko), backend.lib.__name__, str(backend.cls))
        )
    elif len(ko) == 1:
        raise RuntimeError(
            "Parameter `%s` is not consistent with the provided backend. Since\n"
            "`backend.lib` is `%s`, `%s` must be a %s instance."
            % (ko[0], backend.lib.__name__, ko[0], str(backend.cls))
        )
    
    return True

def _check_seq_(t=None, dtype=None, n=None, ndim=None, allow_array_like=False, **kwargs):
    """Perform consistency checks for sequence kwargs.
    
    Parameters
    ----------
    
    t : <class 'type'>, optional    
        if given, check that each non None item in ``kwargs`` is a
        sequence of elements with type `t`.
    
    dtype :  <class 'type'> or <class 'torch.type'>, optional
        if given, check that each non None item in ``kwargs`` is a
        sequence of array_like with datatype `dtype`.
    
    n : int, optional 
        if given, check that each non None item in ``kwargs`` is a
        sequence with length `n`.
    
    ndim : int, optional 
        if given, each non None item in ``kwargs`` is assumed to be an
        array_like and we check that those array_like elements have a
        number of dimensions equal to ``ndim``.
    
    allow_array_like: boolean,
        when set to True, we allow kwargs to be array_like (instead of
        sequences) and perform the tests described above as if they
        were sequences.
    
    Return
    ------
    
    no_error_flag : bool
        True if the test is successful (otherwise an exception is
        raised).
    
    """
    for key, seq in kwargs.items():
        
        if seq is not None:
            
            # check wether seq is a sequence or an array_like
            array_like = hasattr(seq, "shape") and hasattr(seq, "__getitem__")
            
            # check seq
            if not isinstance(seq, (tuple, list)):
                if not allow_array_like:
                    raise RuntimeError(
                        f"Parameter `{key}` must be a sequence (= tuple or list)"
                    )
                elif not array_like:
                    raise RuntimeError(
                        f"Parameter `{key}` must be a sequence (= tuple or list) or an array_like"
                    )
            
            # check n (if given)
            if n is not None and n != len(seq):
                raise RuntimeError(            
                    f"Parameter `{key}` must have len equal to {n}"
                )
            
            # check items type (if given)
            if t is not None:
                if array_like and not isinstance(seq, t):                   
                    raise RuntimeError(
                        f"Parameter `{key}` must have type {t}"
                    )
                elif not all([isinstance(item, t) for item in seq]):
                    raise RuntimeError(
                        f"All elements in `{key}` must have type {t}"
                    )
            
            # check items dtype (if given)
            if dtype is not None:
                if array_like and dtype != seq.dtype:
                    raise RuntimeError(
                        f"Parameter `{key}` must have dtype {dtype}"
                    )
                elif not all([dtype == item.dtype for item in seq]):
                    raise RuntimeError(
                        f"All elements in `{key}` must have dtype {dtype}"
                    )
            
            # check ndim (if given)
            if ndim is not None:
                if array_like and seq.ndim != ndim + 1:                    
                    raise RuntimeError(
                        f"The parameter `{key}` must satisfy {key}.ndim == {ndim + 1}"
                    )
                elif not all([ndim == item.ndim for item in seq]):
                    raise RuntimeError(
                        f"All elements in `{key}` must have a number of dimensions equal to {ndim}"
                    )
    
    return True

def _check_seq_of_seq_(t=None, dtype=None, len0=None, len1=None, len2=None, ndim=None, tlen0=None, allow_array_like=False, **kwargs):
    """Perform consistency checks for sequence of sequence kwargs.
    
    Parameters
    ----------
    
    t : <class 'type'>, optional
        if given, check that each non None item in ``kwargs`` is a
        sequence of sequence(s) of elements with type `t`.
    
    dtype :  <class 'type'> or <class 'torch.type'>, optional
        if given, check that each non None item in ``kwargs`` is a
        sequence of sequence(s) of array_like with datatype `dtype`.
    
    len0 : int, optional 
        if given, check that each non None item in ``kwargs`` has
        length `len0`.
    
    len1 : int, optional 
        if given, check that each non None item in ``kwargs`` is a
        sequence made of sequence(s) with length `len1`.
    
    len2 : int, optional 
        if given, check that each non None leaf in ``kwargs`` has
        length `len2`.
    
    tlen0 : sequence of int, optional
        if given, check that each non None item in ``kwargs`` has
        length in `tlen0`.
    
    ndim : int, optional
        if given, check that each non None leaf in ``kwargs`` has
        a number of dimensions equal to `ndim`.
    
    allow_array_like: boolean,
        when set to True, we allow kwargs to be array_like (instead of
        sequences) and perform the tests described above as if they
        were sequences.
    
    Return
    ------
    
    no_error_flag : bool
        True if the test is successful (otherwise an exception is
        raised).

    """
    for key, seq in kwargs.items():
        
        if seq is not None:
            
            # check wether seq is a sequence of sequences or an array_like
            array_like = hasattr(seq, "shape") and hasattr(seq, "__getitem__")
            seqofseq = isinstance(seq, (tuple, list)) and all((isinstance(s, (tuple, list)) for s in seq))
            
            # check seq of seq & flatten
            if not seqofseq:
                if not allow_array_like:
                    raise RuntimeError(
                        f"Parameter `{key}` must be a sequence of sequence(s)"
                    )
                elif not array_like:
                    raise RuntimeError(
                        f"Parameter `{key}` must be a sequence of sequence(s) or an array_like"
                    )
            
            # check len0 (if given)
            if len0 is not None and len0 != len(seq):
                raise RuntimeError(
                    f"Parameter `{key}` must satisfy len({key}) = {len0}"
                )
            
            # check len1 (if given)
            if len1 is not None:
                if array_like and not seq.shape[1] == len1:
                    raise RuntimeError(
                        f"Parameter `{key}` must satisfy {key}.shape[1] = {len1}"
                    )
                elif not all((len1 == len(s) for s in seq)):
                    raise RuntimeError(
                        f"All elements in `{key}` must have length {len1}"
                    )
            
            # check len2 (if given)
            if len2 is not None:
                if array_like and not seq.shape[2] == len2:
                    raise RuntimeError(
                        f"Parameter `{key}` must satisfy {key}.shape[2] = {len2}"
                    )
                elif not all((len2 == len(leaf) for subseq in seq for leaf in subseq)):
                    raise RuntimeError(
                        f"All leaf elements in `{key}` must have length {len2}"
                    )
            
            # check tlen0 (if given)
            if tlen0 is not None:
                if array_like and seq.shape[0] not in tlen0:
                    raise RuntimeError(
                        f"Parameter `{key}` must satisfy {key}.shape[0] in {tlen0}"
                    )
                elif len(seq) not in tlen0:
                    raise RuntimeError(            
                        f"The sequence of sequence(s) parameter `{key}` must have length in {tlen0}"
                    )
            
            # check ndim (if given)
            if ndim is not None:
                if array_like and not seq.ndim == ndim + 2:
                    print(f"here, ndim={ndim}")
                    raise RuntimeError(
                        f"Parameter `{key}` must satisfy {key}.ndim == {ndim + 2}"
                    )
                elif not all((ndim == leaf.ndim for subseq in seq for leaf in subseq)):
                    raise RuntimeError(
                        f"All leaf elements in `{key}` must have a number of dimensions equal to {ndim}"
                    )
            
            # check leaves type (if given)
            if t is not None:
                if array_like and not isinstance(seq, t):
                    raise RuntimeError(
                        f"Parameter `{key}` must have type {t}"
                    )
                elif not all([isinstance(leaf, t) for subseq in seq for leaf in subseq]):
                    raise RuntimeError(
                        f"All leaf elements in `{key}` must have type {t}"
                    )
            
            # check leaves dtype (if given)
            if dtype is not None:
                if array_like and not seq.dtype == dtype:
                    raise RuntimeError(
                        f"Parameter `{key}` must have dtype {dtype}"
                    )
                elif not all([dtype == leaf.dtype for subseq in seq for leaf in subseq]):
                    raise RuntimeError(
                        f"All leaf elements in `{key}` must have dtype {dtype}"
                    )
    
    return True

def _is_array_like_(x):
    """check wether or not x is an array_like"""
    return hasattr(x, "shape") and hasattr(x, "__getitem__")

def _max_len_(**kwargs):
    """Compute max length of all elements in kwargs (return None if no element in ``kwargs`` is neither a sequence or an array_like).
    
    """
    L = [len(value) if isinstance(value, (tuple, list)) or _is_array_like_(value) else -1 for key, value in kwargs.items()]
    max_L = max(L)
    out = None if max_L == -1 else max_L
    return out

def _backend_inference_(**kwargs):
    """Return a backend inferred from a sequence of array_like inputs.

    """
    if len(kwargs) > 0:
        
        # retrieve type of the first input
        #first_input = kwargs[tuple(kwargs.keys())[0]]
        first_input = kwargs[next(iter(kwargs))]
        cls = type(first_input)
        module = cls.__module__
        classname = cls.__name__
        
        # check type consistency with other inputs
        ko = _check_(lambda x : isinstance(x, cls), **kwargs)
        if len(ko) > 0:
            raise RuntimeError(
                "Backend inference failed, parameters %s are not type consistent.\n"
                "Those parameter must have the same type (numpy.ndarray, "
                "cupy.ndarray or torch.Tensor)." % (str(tuple(kwargs.keys())))
            )

        # all inputs must be ndarray or Tensor
        if classname not in ('ndarray', 'Tensor'):
            raise RuntimeError(
                "Backend inference failed, parameters %s must be all\n"
                "numpy.ndarray or cupy.ndarray or torch.Tensor arrays"
                % str(tuple(kwargs.keys()))
            )
        
        # create a numpy or cupy or torch backend instance
        import pyepri.backends as backends
        if 'numpy' == module:
            backend = backends.create_numpy_backend()
        elif 'cupy' == module:
            backend = backends.create_cupy_backend()
        elif 'torch' == module:
            #device = str(first_input.device).split(':')[0]
            device = first_input.device.type
            ko = _check_(lambda x : device == x.device.type, **kwargs)
            if len(ko) > 0:
                raise RuntimeError(
                    "Backend inference failed, device inconsistency for parameter(s) %s.\n"
                    % (str(ko))
                )
            backend = backends.create_torch_backend(device)
        else:
            raise RuntimeError(
                "Backend inference failed, unsupported module for parameter(s) %s."
                % (str(tuple(kwargs.keys())))
            )
    else:
        raise RuntimeError(
            "Backend inference failed (at least one input must be provided)"
        )
    
    return backend

