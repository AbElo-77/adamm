from .store import ProvenanceStore

class ProvenanceAudit:
    def __init__(self, store: ProvenanceStore):
        self.store = store

    def trace_ancestors(self, fp: str) -> list[str]:
        ancestors = []
        lineage = self.store.get_lineage(fp)
        if lineage:
            for parent_fp in lineage.parents:
                ancestors.append(parent_fp)
                ancestors.extend(self.trace_ancestors(parent_fp))
        return ancestors

    def trace_descendants(self, fp: str) -> list[str]:
        descendants = []
        lineage = self.store.get_lineage(fp)
        if lineage:
            for child_fp in lineage.children:
                descendants.append(child_fp)
                descendants.extend(self.trace_descendants(child_fp))
        return descendants


def failed_runs(store: ProvenanceStore) -> list[str]:
    failed = []
    for fp in store.lineage.keys():
        meta = store.get_metadata(fp)
        if "exit_code" in meta and int(meta["exit_code"]) != 0:
            failed.append(fp)
    return failed
