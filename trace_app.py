import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name != '<module>':
            print(f'CALL: {func_name}')
    return trace_calls

sys.settrace(trace_calls)

try:
    import app
except BaseException as e:
    pass
