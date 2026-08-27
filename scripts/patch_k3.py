import os

path = "/mnt/d/hamza/kimi-k3-in-c/src/cli/k3_run.c"
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    target_old = """static double mem_available_bytes(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double kb = 0.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemAvailable:", 13)) { kb = atof(line + 13); break; }
    fclose(f);
    return kb * 1024.0;
}"""

    # in case already partially replaced
    import re
    func_pattern = r"static double mem_available_bytes\(void\)[\s\S]*?return kb \* 1024\.0;\s*\}"
    clean_func = """static double mem_available_bytes(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double kb = 0.0;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "MemAvailable:", 13)) { kb += atof(line + 13); }
        if (!strncmp(line, "SwapFree:", 9)) { kb += atof(line + 9); }
    }
    fclose(f);
    return kb * 1024.0;
}"""

    new_code = re.sub(func_pattern, clean_func, code, count=1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_code)
    print("Function cleanly updated with while block braces.")
