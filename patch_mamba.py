with open('/opt/dlami/nvme/mamba/setup.py', 'r') as f:
    content = f.read()

import re
pattern = r'cc_flag\.append\("-gencode"\)[\s\S]*?(?=# HACK: The compiler flag)'
replacement = 'cc_flag = ["-gencode", "arch=compute_86,code=sm_86"]\n\n    '

new_content = re.sub(pattern, replacement, content)
with open('/opt/dlami/nvme/mamba/setup.py', 'w') as f:
    f.write(new_content)

print("setup.py successfully patched for sm_86 architecture!")
