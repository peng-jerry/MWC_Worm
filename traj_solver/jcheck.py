import sys, numpy as np, io, contextlib, textwrap

scenario = sys.argv[1]
sys.argv = ['animate_transition.py', scenario]

import animate_transition as _at
captured = []
_at.render_video = lambda frames, *a, **kw: captured.extend(frames)

with open('animate_transition.py') as f:
    src = f.read()
main_src = textwrap.dedent(src.split('if __name__ == "__main__":')[1])

globs = dict(vars(_at))
globs['__name__'] = '__main__'

with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(main_src, 'animate_transition.py', 'exec'), globs)

prev = None; found = 0; mx = 0
for i, (q, info, p) in enumerate(captured):
    if prev is not None:
        dq = np.degrees(np.abs((q - prev + np.pi) % (2*np.pi) - np.pi))
        mdq = float(np.max(dq))
        if mdq > 8:
            found += 1
            print(f'  frame {i+1:3d}  s={i/89:.3f}  max_dq={mdq:.1f}')
        mx = max(mx, mdq)
    prev = q.copy()
print(f'RESULT  scenario={scenario}  jumps>8={found}  max_jump={mx:.1f}deg')
