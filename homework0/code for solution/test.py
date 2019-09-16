from addition import add


print("the result of add(1,2)")
print(add(1,2))

print ('\n')
print ('\n')

from buyLotsOfFruit import buyLotsOfFruit

orderList = [ ('apples', 2.0), ('pears', 3.0), ('limes', 4.0), ('strawberries', 5.0) ]
print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))

print ('\n')

orderList = [ ('apples', 2.0), ('pears', 3.0), ('limes', 4.0), ('lemmon', 5.0) ]
print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))


print ('\n')
print ('\n')


from shopSmart import shopSmart
import shop
orders = [('apples',1.0), ('oranges',3.0)]
dir1 = {'apples': 2.0, 'oranges':1.0}
shop1 =  shop.FruitShop('shop1',dir1)
dir2 = {'apples': 1.0, 'oranges': 5.0}
shop2 = shop.FruitShop('shop2',dir2)
shops = [shop1, shop2]
print("For orders ", orders, ", the best shop is", shopSmart(orders, shops).getName())
orders = [('apples',3.0)]
print("For orders: ", orders, ", the best shop is", shopSmart(orders, shops).getName())

print ('\n')
print ('\n')