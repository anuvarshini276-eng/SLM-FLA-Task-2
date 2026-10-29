"""Exact recognizer for L = {w in {0,1}* : w ends in 0}."""
MAX_LENGTH = 256


def validate(text):
    if not isinstance(text, str):
        raise ValueError('string must be a JSON string.')
    if len(text) > MAX_LENGTH:
        raise ValueError(f'Use at most {MAX_LENGTH} characters.')
    if any(c not in '01' for c in text):
        raise ValueError('Only 0 and 1 are allowed; spaces are not removed.')


def run_dfa(text):
    validate(text)
    state = 'q0'
    trace = [{'step': 0, 'symbol': None, 'state': state}]
    for i, symbol in enumerate(text, 1):
        state = 'q1' if symbol == '0' else 'q0'
        trace.append({'step': i, 'symbol': symbol, 'state': state})
    return {'accepted': state == 'q1', 'final_state': state, 'trace': trace}
