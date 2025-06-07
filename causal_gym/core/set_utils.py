import pydash


class SetUtils():

    @staticmethod
    def union(a, b, iteratee=None):
        if iteratee is None:
            return pydash.arrays.union(a, b)
        else:
            return pydash.arrays.union_by(a, b, iteratee=iteratee)

    @staticmethod
    def difference(a, b, iteratee=None):
        if iteratee is None:
            return pydash.arrays.difference(a, b)
        else:
            return pydash.arrays.difference_by(a, b, iteratee=iteratee)

    @staticmethod
    def intersection(a, b, iteratee=None):
        if iteratee is None:
            return pydash.arrays.intersection(a, b)
        else:
            return pydash.arrays.intersection_by(a, b, iteratee=iteratee)

    @staticmethod
    def belongs(a, b, comparator=None):
        if comparator is not None:
            for item in b:
                if comparator(a, item):
                    return True

            return False
        elif isinstance(a, list):
            for item in b:
                if SetUtils.isEqual(a, item):
                    return True

            return False
        else:
            return pydash.collections.includes(b, a)

    @staticmethod
    def isEmpty(arr):
        return pydash.predicates.is_empty(arr)

    @staticmethod
    def isSubset(sub, set, iteratee=None):
        return len(SetUtils.difference(sub, set, iteratee)) == 0

    @staticmethod
    def equals(a, b, iteratee=None):
        if a is None and b is None:
            return True

        return SetUtils.isSubset(a, b, iteratee) and SetUtils.isSubset(b, a, iteratee)

    @staticmethod
    def isEqual(a, b):
        if a is None and b is None:
            return True

        return pydash.predicates.is_equal(a, b)

    @staticmethod
    def unique(arr):
        return pydash.arrays.uniq(arr)

    @staticmethod
    def uniqWith(array, comparator=None):
        return pydash.arrays.uniq_with(array, comparator)