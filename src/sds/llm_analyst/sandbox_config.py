"""Rules and limits of the safe execution of generated code.

The lists are a first barrier, not a wall: the real protection is that the child process receives only the
synthetic data and no secret.
"""

# Imports the generated code may write (exact names: a submodule such as numpy.ctypeslib is not allowed).
ALLOWED_MODULES = frozenset({"pandas", "numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.ticker"})

# Built-in functions refused by the static check and missing in the child: code execution, files, input,
# introspection.
FORBIDDEN_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "input", "breakpoint", "help", "exit", "quit",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr", "memoryview",
})

# Attributes refused by the static check: files and network, string evaluation, ways out to the system and
# stack frames (which lead back to the real built-ins).
FORBIDDEN_ATTRIBUTES = frozenset({
    "to_csv", "to_excel", "to_json", "to_parquet", "to_pickle", "to_hdf", "to_sql", "to_feather", "to_stata",
    "to_xml", "to_html", "to_latex", "to_markdown", "to_clipboard", "to_orc",
    "savefig", "imsave", "imread", "tofile", "fromfile", "load", "save", "savez", "savez_compressed", "savetxt",
    "loadtxt", "genfromtxt", "fromregex", "memmap",
    "eval", "query",
    "os", "sys", "subprocess", "builtins", "importlib", "ctypes", "ctypeslib", "system", "popen",
    "gi_frame", "cr_frame", "ag_frame", "tb_frame", "tb_next", "f_back", "f_globals", "f_locals", "f_builtins",
    "f_code",
})
FORBIDDEN_ATTRIBUTE_PREFIXES = ("read_",)  # pandas readers: files, URLs, pickles

# The child's environment: only what Python needs to start (SYSTEMROOT on Windows), never a key.
PASSED_ENV = ("PATH", "SYSTEMROOT")

RUN_TIMEOUT_SECONDS = 30   # start (~0.75 s with the font cache ready) + reading the data + the chart
MAX_SUMMARY_ROWS = 30      # the summary table holds the numbers of one chart: a few groups, not the data
MAX_SUMMARY_COLUMNS = 10
CHART_DPI = 100
