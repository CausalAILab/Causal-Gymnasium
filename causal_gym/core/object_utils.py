# import * as _ from "lodash";
import copy
import itertools

class ObjectUtils():

    @staticmethod
    def diff(a, b):
        return

#     static diff(a, b): any {
#         function changes(object, base) {
#             return _.transform(object, function (result, value, key) {
#                 if (!_.isEqual(value, base[key]))
#                     result[key] = (_.isObject(value) && _.isObject(base[key])) ? changes(value, base[key]) : value;
#             });
#         }
#         return changes(a, b);
#     }

    @staticmethod
    def equals(a, b):
        return False

#     static equals(a, b): boolean {
#         return _.isEqual(a, b);
#     }

#     static merge(a, b) {
#         _.merge(a, b);
#     }

#     static pick(a, b) {
#         return _.pick(a, _.keys(b));
#     }

#     static omit(a, b) {
#         return _.omit(a, _.keys(b));
#     }

    @staticmethod
    def clone(a, deep = True):
        return copy.deepcopy(a) if deep else copy.copy(a)

#     static get(a, path) {
#         return _.get(a, path);
#     }

#     static set(a, path, value) {
#         _.set(a, path, value);
#     }

#     static keys(a) {
#         return _.keys(a);
#     }

#     static getKeys(a) {
#         const keyify = (obj, prefix = '') =>
#             Object.keys(obj).reduce((res, el) => {
#                 if (Array.isArray(obj[el])) {
#                     return res;
#                 } else if (typeof obj[el] === 'object' && obj[el] !== null) {
#                     return [...res, ...keyify(obj[el], prefix + el + '.')];
#                 } else {
#                     return [...res, prefix + el];
#                 }
#             }, []);

#         return keyify(a);
#     }


    # symmetric if true, symmetric pairs will be outputted (e.g., X,Y and Y,X). defaults to false.
    @staticmethod
    def pairs(array, symmetric = False):
        if symmetric:
            return list(itertools.permutations(array, 2))
        else:
            return list(itertools.combinations(array, 2))


#     // https://github.com/SeregPie/lodash.combinations#readme
#     /**
#      * 
#      * @param values 
#      * @param n the length of each resulting combination
#      */
#     static combinations(values, n: number) {
#         let rr = function (func) {
#             let recur = function (...args) {
#                 return func.call(this, recur, ...args);
#             };
#             return recur;
#         }

#         values = _.values(values);
#         let combinations = [];

#         rr((recur, combination, index) => {
#             if (combination.length < n) {
#                 _.find(values, (value, index) => {
#                     recur(_.concat(combination, [value]), index + 1);
#                 }, index);
#             } else {
#                 combinations.push(combination);
#             }
#         })([], 0);

#         return combinations;
#     }

#     // https://codereview.stackexchange.com/questions/7001/generating-all-combinations-of-an-array
#     static combinationsOfArrays(values: any[][]): any[][] {
#         if (values.length == 0)
#             return [];
#         else if (values.length == 1)
#             return values[0];

#         let result = [];

#         let allCasesOfRest: any[] = this.combinationsOfArrays(values.slice(1));  // recur with the rest of array

#         for (let c in allCasesOfRest) {
#             for (var i = 0; i < values[0].length; i++) {
#                 let a: any = values[0][i];
#                 let b: any[] = allCasesOfRest[c];
#                 result.push([a].concat(b));
#             }
#         }

#         return result;
#     }

    @staticmethod
    def makeArray(elements):
        if not isinstance(elements, list):
            elements = [elements]
        
        return elements

#     /**
#      * If the given parameter is not an array it will be wrapped inside one. If
#      * the parameter is an array the same will be returned
#      * @param elements A single object or array
#      * @returns The same array if an array is given or a new array containing the
#      * single element if one is given
#      */
#     static makeArray<T>(elements: T[] | T): T[] {
#         if (!(elements instanceof Array))
#             elements = [elements];
#         return elements;
#     }
# }