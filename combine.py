import os
import subprocess
from datetime import datetime

license_boilerplate = """/*
 * Kernel Float: Header-only library for vector types and reduced precision floating-point math.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

"""

directory = "include/kernel_float"
contents = dict()

for filename in os.listdir(directory):
    if filename.endswith(".h"):
        with open(f"{directory}/{filename}") as handle:
            print(f"reading {filename}")
            contents[filename] = handle.read()

deps = dict((filename, []) for filename in contents)
for dep in contents:
    for filename in contents:
        pattern = f'#include "{dep}"\n'

        if pattern in contents[filename]:
            deps[filename].append(dep)
            contents[filename] = contents[filename].replace(pattern, "\n")

date = datetime.now()
git_hash = "???"

try:
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
except Exception as e:
    print(f"warning: {e}")

output = license_boilerplate
output += "\n".join([
    "//" + "=" * 80,
    "// this file has been auto-generated, do not modify its contents!",
    f"// date: {date}",
    f"// git hash: {git_hash}",
    "//" + "=" * 80,
    "",
    "",
])

while len(deps):
    key = min(k for k in deps if len(deps[k]) == 0)

    for k in deps:
        if key in deps[k]:
            deps[k].remove(key)

    del deps[key]

    print(f"writing {key}")
    output += contents[key].strip() + "\n"

with open("single_include/kernel_float.h", "w") as handle:
    handle.write(output)

