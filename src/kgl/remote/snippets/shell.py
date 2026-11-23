import codecs, subprocess, sys, os
cwd = __CWD__
p = subprocess.Popen(__CMD__, shell=True, executable="/bin/bash", stdout=subprocess.PIPE,
                     stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                     cwd=cwd if os.path.isdir(cwd) else None)
decode = codecs.getincrementaldecoder("utf-8")("replace")
for chunk in iter(lambda: p.stdout.read1(4096), b""):
    sys.stdout.write(decode.decode(chunk)); sys.stdout.flush()
sys.stdout.write(decode.decode(b"", final=True))
p.wait()
sys.stdout.write("\n"); _emit({"rc": p.returncode})
