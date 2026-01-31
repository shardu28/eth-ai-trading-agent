"""
Trade Reconciliation Module
Syncs exchange order history and fills with local tradebook.csv

This is the CRITICAL module that ensures:
1. Local state matches exchange state
2. TP/SL hits are detected and recorded
3. No silent data corruption
4. Deterministic state recovery

Called after every trade execution in main.py
"""

import logging
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TradeReconciler:
    """
    Reconciles exchange state with tradebook.csv
    
    Critical responsibilities:
    - Detect filled orders
    - Detect TP/SL triggers
    - Update tradebook with actual exchange data
    - Prevent state divergence
    """
    
    def __init__(self, api_client, tradebook_path: str = "tradebook.csv"):
        """
        Initialize reconciler
        
        Args:
            api_client: DeltaExchangeAPI instance
            tradebook_path: Path to tradebook.csv
        """
        self.api = api_client
        self.tradebook_path = Path(tradebook_path)
        
        # Ensure tradebook exists
        self._init_tradebook()
        
        logger.info(f"Trade reconciler initialized with {tradebook_path}")
    
    def _init_tradebook(self):
        """Initialize tradebook.csv if it doesn't exist"""
        if not self.tradebook_path.exists():
            logger.info("Creating new tradebook.csv")
            with open(self.tradebook_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'signal_time',
                    'product_id',
                    'product_symbol',
                    'entry_order_id',
                    'entry_client_order_id',
                    'side',
                    'size',
                    'entry_price_intended',
                    'entry_price_actual',
                    'entry_fill_time',
                    'sl_price',
                    'tp_price',
                    'sl_order_id',
                    'tp_order_id',
                    'exit_order_id',
                    'exit_price',
                    'exit_fill_time',
                    'exit_reason',  # tp_hit, sl_hit, manual, session_end
                    'pnl',
                    'commission_paid',
                    'status',  # open, closed, cancelled, error
                    'error_message'
                ])
    
    def reconcile_after_entry(self, client_order_id: str, product_id: int,
                            intended_entry_price: str, sl_price: str, tp_price: str,
                            side: str, size: int, signal_time: int) -> Tuple[bool, Dict]:
        """
        Reconcile immediately after entry order placement
        
        This function:
        1. Verifies entry order was accepted
        2. Gets actual fill price (if filled)
        3. Records entry in tradebook
        4. Returns trade record for TP/SL order placement
        
        Args:
            client_order_id: Client order ID used for entry
            product_id: Product ID
            intended_entry_price: Intended entry price
            sl_price: Stop loss price
            tp_price: Take profit price
            side: "buy" or "sell"
            size: Position size
            signal_time: Signal timestamp (microseconds)
        
        Returns:
            (success: bool, trade_record: dict)
        """
        logger.info(f"Reconciling entry order: {client_order_id}")
        
        # Wait briefly for exchange processing
        time.sleep(0.5)
        
        # Get order by client_order_id
        success, orders = self.api.get_active_orders(product_id=product_id)
        
        if not success:
            logger.error(f"Failed to get active orders: {orders}")
            return False, {}
        
        # Find our order
        entry_order = None
        for order in orders:
            if order.get('client_order_id') == client_order_id:
                entry_order = order
                break
        
        if not entry_order:
            # Order might be filled already, check history
            logger.info("Order not in active orders, checking history")
            success, history = self._get_recent_order_history(product_id, client_order_id)
            
            if success and history:
                entry_order = history[0]
            else:
                logger.error(f"Entry order {client_order_id} not found")
                return False, {}
        
        # Extract order details
        order_id = entry_order.get('id')
        state = entry_order.get('state')
        unfilled_size = entry_order.get('unfilled_size', size)
        
        logger.info(f"Entry order state: {state}, unfilled: {unfilled_size}/{size}")
        
        # Get actual fill price if filled
        actual_entry_price = None
        entry_fill_time = None
        commission_paid = 0.0
        
        if unfilled_size == 0 or state == 'closed':
            # Order fully filled, get fill details
            success, fills = self._get_order_fills(order_id, product_id)
            
            if success and fills:
                # Calculate weighted average fill price
                total_value = 0
                total_size = 0
                total_commission = 0
                
                for fill in fills:
                    fill_price = float(fill['price'])
                    fill_size = fill['size']
                    fill_commission = float(fill['commission'])
                    
                    total_value += fill_price * fill_size
                    total_size += fill_size
                    total_commission += abs(fill_commission)
                    
                    if not entry_fill_time:
                        entry_fill_time = fill.get('created_at')
                
                actual_entry_price = str(total_value / total_size) if total_size > 0 else intended_entry_price
                commission_paid = total_commission
                
                logger.info(f"Entry filled @ {actual_entry_price}, commission: {commission_paid}")
        
        # Create trade record
        trade_record = {
            'timestamp': int(time.time() * 1000000),
            'signal_time': signal_time,
            'product_id': product_id,
            'product_symbol': self._get_product_symbol(product_id),
            'entry_order_id': order_id,
            'entry_client_order_id': client_order_id,
            'side': side,
            'size': size,
            'entry_price_intended': intended_entry_price,
            'entry_price_actual': actual_entry_price or '',
            'entry_fill_time': entry_fill_time or '',
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_order_id': '',
            'tp_order_id': '',
            'exit_order_id': '',
            'exit_price': '',
            'exit_fill_time': '',
            'exit_reason': '',
            'pnl': '',
            'commission_paid': commission_paid,
            'status': 'open' if actual_entry_price else 'pending',
            'error_message': ''
        }
        
        # Write to tradebook
        self._append_to_tradebook(trade_record)
        
        logger.info(f"Trade record created: status={trade_record['status']}")
        
        return True, trade_record
    
    def reconcile_exit(self, entry_client_order_id: str, product_id: int) -> Tuple[bool, Dict]:
        """
        Reconcile after position exit (TP hit, SL hit, or manual close)
        
        This function:
        1. Finds the original trade in tradebook
        2. Gets exit fill details from exchange
        3. Calculates PnL
        4. Updates tradebook with exit data
        
        Args:
            entry_client_order_id: Client order ID of entry
            product_id: Product ID
        
        Returns:
            (success: bool, updated_record: dict)
        """
        logger.info(f"Reconciling exit for trade: {entry_client_order_id}")
        
        # Find trade record in tradebook
        trade_record = self._find_trade_in_tradebook(entry_client_order_id)
        
        if not trade_record:
            logger.error(f"Trade {entry_client_order_id} not found in tradebook")
            return False, {}
        
        # Get recent fills for this product
        end_time = int(time.time() * 1000000)
        start_time = int(trade_record['entry_fill_time']) if trade_record['entry_fill_time'] else end_time - (3600 * 1000000)
        
        success, fills = self.api.get_fills(
            product_id=product_id,
            start_time=start_time,
            end_time=end_time,
            page_size=50
        )
        
        if not success:
            logger.error(f"Failed to get fills: {fills}")
            return False, trade_record
        
        # Find exit fills (opposite side of entry)
        entry_side = trade_record['side']
        exit_side = 'sell' if entry_side == 'buy' else 'buy'
        
        exit_fills = []
        for fill in fills:
            if fill['side'] == exit_side:
                exit_fills.append(fill)
        
        if not exit_fills:
            logger.warning("No exit fills found")
            return False, trade_record
        
        # Calculate exit details
        total_value = 0
        total_size = 0
        total_commission = 0
        exit_fill_time = None
        exit_order_id = None
        
        for fill in exit_fills:
            fill_price = float(fill['price'])
            fill_size = fill['size']
            fill_commission = float(fill['commission'])
            
            total_value += fill_price * fill_size
            total_size += fill_size
            total_commission += abs(fill_commission)
            
            if not exit_fill_time:
                exit_fill_time = fill.get('created_at')
                exit_order_id = fill.get('order_id')
        
        exit_price = total_value / total_size if total_size > 0 else 0
        
        # Calculate PnL
        entry_price = float(trade_record['entry_price_actual'])
        size = int(trade_record['size'])
        
        if entry_side == 'buy':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        # Subtract commissions
        entry_commission = float(trade_record['commission_paid'])
        total_commission += entry_commission
        pnl -= total_commission
        
        # Determine exit reason
        exit_reason = self._determine_exit_reason(
            exit_price, 
            float(trade_record['sl_price']), 
            float(trade_record['tp_price']),
            entry_side
        )
        
        logger.info(f"Exit @ {exit_price:.4f}, PnL: {pnl:.4f}, Reason: {exit_reason}")
        
        # Update trade record
        trade_record.update({
            'exit_order_id': exit_order_id,
            'exit_price': str(exit_price),
            'exit_fill_time': exit_fill_time,
            'exit_reason': exit_reason,
            'pnl': str(pnl),
            'commission_paid': str(total_commission),
            'status': 'closed'
        })
        
        # Update tradebook
        self._update_trade_in_tradebook(entry_client_order_id, trade_record)
        
        return True, trade_record
    
    def _get_recent_order_history(self, product_id: int, client_order_id: str,
                                  lookback_hours: int = 1) -> Tuple[bool, List]:
        """Get recent order history and find specific order"""
        end_time = int(time.time() * 1000000)
        start_time = end_time - (lookback_hours * 3600 * 1000000)
        
        success, history = self.api.get_order_history(
            product_id=product_id,
            start_time=start_time,
            end_time=end_time,
            page_size=50
        )
        
        if not success:
            return False, []
        
        matching_orders = [o for o in history if o.get('client_order_id') == client_order_id]
        return True, matching_orders
    
    def _get_order_fills(self, order_id: int, product_id: int) -> Tuple[bool, List]:
        """Get fills for a specific order"""
        # Get recent fills
        end_time = int(time.time() * 1000000)
        start_time = end_time - (3600 * 1000000)  # Last hour
        
        success, fills = self.api.get_fills(
            product_id=product_id,
            start_time=start_time,
            end_time=end_time,
            page_size=100
        )
        
        if not success:
            return False, []
        
        # Filter by order_id
        order_fills = [f for f in fills if f['order_id'] == str(order_id)]
        return True, order_fills
    
    def _determine_exit_reason(self, exit_price: float, sl_price: float, 
                               tp_price: float, entry_side: str) -> str:
        """
        Determine why position was exited
        
        Logic:
        - If exit price near SL → sl_hit
        - If exit price near TP → tp_hit
        - Otherwise → manual or session_end (needs external context)
        """
        threshold = 0.001  # 0.1% threshold
        
        if entry_side == 'buy':
            # Long position
            if abs(exit_price - sl_price) / sl_price < threshold:
                return 'sl_hit'
            elif abs(exit_price - tp_price) / tp_price < threshold:
                return 'tp_hit'
        else:
            # Short position
            if abs(exit_price - sl_price) / sl_price < threshold:
                return 'sl_hit'
            elif abs(exit_price - tp_price) / tp_price < threshold:
                return 'tp_hit'
        
        return 'manual'  # Default to manual
    
    def _get_product_symbol(self, product_id: int) -> str:
        """Get product symbol from ID"""
        # Cache this in production
        success, product = self.api.get_product_by_symbol("SOLUSD")
        if success and product.get('id') == product_id:
            return product.get('symbol', 'UNKNOWN')
        return 'UNKNOWN'
    
    def _append_to_tradebook(self, record: Dict):
        """Append new record to tradebook"""
        with open(self.tradebook_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)
        
        logger.info(f"Appended trade to tradebook: {record['entry_client_order_id']}")
    
    def _find_trade_in_tradebook(self, entry_client_order_id: str) -> Optional[Dict]:
        """Find trade record in tradebook"""
        with open(self.tradebook_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['entry_client_order_id'] == entry_client_order_id:
                    return row
        return None
    
    def _update_trade_in_tradebook(self, entry_client_order_id: str, updated_record: Dict):
        """Update existing trade record in tradebook"""
        rows = []
        updated = False
        
        with open(self.tradebook_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            for row in reader:
                if row['entry_client_order_id'] == entry_client_order_id:
                    rows.append(updated_record)
                    updated = True
                else:
                    rows.append(row)
        
        if updated:
            with open(self.tradebook_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Updated trade in tradebook: {entry_client_order_id}")
        else:
            logger.warning(f"Trade not found for update: {entry_client_order_id}")


# ============= USAGE EXAMPLE =============

if __name__ == "__main__":
    """
    Example usage - this will be integrated into main.py
    """
    from api_test import DeltaExchangeAPI
    
    # Setup
    api = DeltaExchangeAPI("config.yml")
    reconciler = TradeReconciler(api, "tradebook.csv")
    
    # After placing entry order in main.py:
    # success, trade_record = reconciler.reconcile_after_entry(
    #     client_order_id="signal_20260131_143022",
    #     product_id=27,
    #     intended_entry_price="150.00",
    #     sl_price="148.00",
    #     tp_price="153.00",
    #     side="buy",
    #     size=1,
    #     signal_time=int(time.time() * 1000000)
    # )
    
    # After position exit (TP hit, SL hit, or manual close):
    # success, updated_record = reconciler.reconcile_exit(
    #     entry_client_order_id="signal_20260131_143022",
    #     product_id=27
    # )
    
    print("Trade reconciliation module ready for integration")
