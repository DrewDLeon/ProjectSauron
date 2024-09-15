try:
    from openvino.runtime import Core  # New API structure in recent versions
except ImportError as e:
    print(e)
