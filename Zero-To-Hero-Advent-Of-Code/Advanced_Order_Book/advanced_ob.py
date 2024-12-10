from enum import Enum
from typing import List, Dict, Optional

class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    STOP_LOSS = 3
    TRAILING_STOP_LOSS = 4

class OrderStatus(Enum):
    NEW = 1
    FILLED = 2
    CANCELLED = 3
    TRIGGERED = 4  # For stop loss and trailing stop loss

class Order:
    def __init__(
        self,
        id: int,
        order_type: OrderType,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        price: Optional[float] = None,  # For market orders, this can be None
        stop_price: Optional[float] = None,  # For stop-loss orders
        trail_amount: Optional[float] = None,  # For trailing stop-loss
        status: OrderStatus = OrderStatus.NEW,
        linked_orders: Optional[List[int]] = None  # For OCO
    ):
        self.id = id
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.trail_amount = trail_amount
        self.status = status
        self.linked_orders = linked_orders

class OrderBook:
    def __init__(self):
        # Initialize orders to be a dictionary of int id's and Order orders
        # next_order_id is a ease of use for order id
        self.orders: Dict[int, Order] = {}
        self.next_order_id = 0

    # Helper function to increment the id
    def _generate_order_id(self) -> int:
        self.next_order_id += 1
        return self.next_order_id

    # Place an order by calling Order with the parameters, use **kwargs for arguments when we might not need them to avoid errors
    # Then put it in the list with its order id
    def place_order(self, order_type: OrderType, side: str, quantity: float, **kwargs) -> int:
        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            order_type=order_type,
            side=side,
            quantity=quantity,
            **kwargs
        )
        self.orders[order_id] = order
        return order_id

    # Set order status to canceled for an indidivdual order and its linked on if applicable
    def cancel_order(self, order_id: int):
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.CANCELLED
            # Cancel linked orders if OCO
            if order.linked_orders:
                for linked_order_id in order.linked_orders:
                    self.cancel_order(linked_order_id)

    def execute_trade(self, side: str, quantity: float, best_bid: float, best_ask: float):
        # Best price execution
        price = best_bid if side == 'sell' else best_ask

        # See if we can match any
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
            if order.status != OrderStatus.NEW:
                continue

            if order.side == side:
                # Immediatly attempt to fill a market order
                if order.order_type == OrderType.MARKET:
                    filled_quantity = min(order.quantity, quantity)
                    order.quantity -= filled_quantity
                    if order.quantity == 0:
                        order.status = OrderStatus.FILLED

                elif order.order_type == OrderType.LIMIT:
                    # Must check for price before filling limit
                    if (side == 'buy' and order.price >= price) or (side == 'sell' and order.price <= price):
                        filled_quantity = min(order.quantity, quantity)
                        order.quantity -= filled_quantity
                        if order.quantity == 0:
                            order.status = OrderStatus.FILLED

                elif order.order_type == OrderType.STOP_LOSS:
                    # Change status for stop loss to triggered if at or below
                    if (side == 'buy' and price <= order.stop_price) or (side == 'sell' and price >= order.stop_price):
                        order.status = OrderStatus.TRIGGERED

                elif order.order_type == OrderType.TRAILING_STOP_LOSS:
                    # Appropriately update the trailing stop loss
                    if side == 'buy':
                        if price <= order.stop_price - order.trail_amount:
                            order.status = OrderStatus.TRIGGERED
                    elif side == 'sell':
                        if price >= order.stop_price + order.trail_amount:
                            order.status = OrderStatus.TRIGGERED
                    # Update the stop price for trailing stop loss
                    if side == 'buy':
                        order.stop_price = max(order.stop_price, price - order.trail_amount)
                    elif side == 'sell':
                        order.stop_price = min(order.stop_price, price + order.trail_amount)

            if quantity <= 0:
                break

    # Link orders by their id
    def oco_order(self, order1: int, order2: int):
        if order1 in self.orders and order2 in self.orders:
            self.orders[order1].linked_orders = [order2]
            self.orders[order2].linked_orders = [order1]

    # Return order status from it's id
    def get_order_status(self, order_id: int) -> OrderStatus:
        return self.orders[order_id].status if order_id in self.orders else None
    
    # to_string for visualizaiton
    def to_string(self) -> str:
        result = []
        for _, order in self.orders.items():
            order_info = (f"Order ID: {order.id}, Type: {order.order_type.name}, Side: {order.side}, "
                f"Quantity: {order.quantity}, Price: {order.price}, Stop Price: {order.stop_price}, "
                f"Trail Amount: {order.trail_amount}, Status: {order.status.name}, "
                f"Linked Orders: {order.linked_orders}")
            
            result.append(order_info)
        return "\n".join(result)

# Testing
order_book = OrderBook()
market_order = order_book.place_order(OrderType.MARKET, 'buy', 100)
limit_order = order_book.place_order(OrderType.LIMIT, 'sell', 50, price=20.0)
stop_loss = order_book.place_order(OrderType.STOP_LOSS, 'sell', 50, stop_price=18.0)
trailing_stop = order_book.place_order(OrderType.TRAILING_STOP_LOSS, 'sell', 50, stop_price=20.0, trail_amount=1.0)

# Link orders for OCO (One Cancels Other)
order_book.oco_order(limit_order, stop_loss)

# Simulating some trades
order_book.execute_trade('buy', 150, 19.5, 20.5)  # Best bid 19.5, Best ask 20.5
order_book.execute_trade('sell', 200, 19.5, 20.5)

# Checking status
print(order_book.get_order_status(market_order))
print(order_book.get_order_status(limit_order))
print(order_book.get_order_status(stop_loss))
print(order_book.get_order_status(trailing_stop))
print(order_book.to_string())