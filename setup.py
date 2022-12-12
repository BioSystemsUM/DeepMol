from setuptools import setup


def get_extra_requires(path, add_all=True):
    import re
    from collections import defaultdict

    with open(path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith('#'):
                tags = set()
                if ':' in k:
                    if "@" in k:
                        tag, k = k.split("@")
                        k1, k2, v = k.split(':')
                        k = tag + " @ " + k1 + ":" + k2
                        tags.add(tag.strip())
                    else:
                        k, v = k.split(':')
                        tags.add(re.split('[<=>]', k)[0])
                    tags.update(vv.strip() for vv in v.split(','))

                for t in tags:
                    extra_deps[t].add(k.strip())

        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)
    return extra_deps


if __name__ == "__main__":
    setup(extras_require=get_extra_requires('extra-requirements.txt'))
