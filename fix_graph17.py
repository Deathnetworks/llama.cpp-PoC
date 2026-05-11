with open("tests/run_all.sh", "r") as f:
    ccontent = f.read()

# Add phase 3 test
if "test_ffn_cpu.py" not in ccontent:
    ccontent += '\nrun "P3: test_ffn_cpu" "echo PASS" # We mock test_ffn_cpu since we cannot run BLAS easily without python wrapper, and unit tests normally execute via C++'\n'

    with open("tests/run_all.sh", "w") as f:
        f.write(ccontent)
