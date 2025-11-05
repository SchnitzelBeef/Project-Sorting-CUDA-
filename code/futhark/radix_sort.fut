-- ==
-- compiled input @input.in

import "lib/github.com/diku-dk/sorts/radix_sort"

let sort_u32 : []u32 -> []u32 = radix_sort u32.num_bits u32.get_bit

let main (arr: []u32) : []u32 =
  sort_u32 arr
