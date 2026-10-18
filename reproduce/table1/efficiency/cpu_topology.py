"""Record the physical cores behind each comparison worker's CPU affinity."""
import argparse
import json
from pathlib import Path


def topology(cpus):
    result = {}
    for cpu in cpus:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        result[cpu] = dict(package=int((root / "physical_package_id").read_text()),
                           core=int((root / "core_id").read_text()),
                           siblings=(root / "thread_siblings_list").read_text().strip())
    return result


def independent_cpus(cpus, details=None):
    """Choose one logical CPU from each physical core in the allowed affinity."""
    details = topology(cpus) if details is None else details
    seen = set()
    result = []
    for cpu in sorted(cpus):
        key = (details[cpu]["package"], details[cpu]["core"])
        if key not in seen:
            result.append(cpu)
            seen.add(key)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    tasks = {}
    for task in plan["tasks"]:
        details = topology(task["cpus"])
        physical = sorted(set((info["package"], info["core"]) for info in details.values()))
        tasks[task["name"]] = dict(logical_cpus=task["cpus"], physical_cores=physical, details=details)
    overlaps = []
    names = list(tasks)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            shared = set(map(tuple, tasks[name]["physical_cores"])) & set(map(tuple, tasks[other]["physical_cores"]))
            if shared:
                overlaps.append(dict(workers=[name, other], shared_physical_cores=sorted(shared)))
    print(json.dumps(dict(tasks=tasks, overlaps=overlaps), indent=2))


if __name__ == "__main__":
    main()
