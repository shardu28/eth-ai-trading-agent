"""
Delta Exchange API Test Suite
Production-grade API interaction with comprehensive error handling

This module tests all critical API endpoints for the crypto trading bot.
Zero hallucinations - all logic based on official Delta Exchange API docs.
"""

import hashlib
import hmac
import time
import requests
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DeltaExchangeAPI:
    """
    Delta Exchange API Client
    Implements all authentication, rate limiting, and error handling
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize API client with configuration"""
        self.config = self._load_config(config_path)
        self.base_url = self.config['api']['base_url']
        self.api_key = self.config['api']['api_key']
        self.api_secret = self.config['api']['api_secret']
        
        # Validate credentials
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be configured in config.yml")
        
        # Rate limiting tracking
        self.request_count = 0
        self.window_start = time.time()
        
        logger.info(f"Initialized Delta Exchange API client for {self.base_url}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_signature(self, method: str, timestamp: str, path: str, 
                          query_string: str = "", payload: str = "") -> str:
        """
        Generate HMAC-SHA256 signature for API request
        
        Signature = HMAC-SHA256(secret, method + timestamp + path + query_string + payload)
        Critical: Signature must be generated within 5 seconds of use
        """
        message = method + timestamp + path + query_string + payload
        message_bytes = bytes(message, 'utf-8')
        secret_bytes = bytes(self.api_secret, 'utf-8')
        
        hash_obj = hmac.new(secret_bytes, message_bytes, hashlib.sha256)
        return hash_obj.hexdigest()
    
    def _get_headers(self, method: str, path: str, query_string: str = "", 
                     payload: str = "") -> Dict[str, str]:
        """
        Generate authentication headers for API request
        
        Returns headers dict with:
        - api-key
        - timestamp
        - signature
        - User-Agent (required to avoid 403)
        - Content-Type
        """
        timestamp = str(int(time.time()))
        signature = self._generate_signature(method, timestamp, path, query_string, payload)
        
        return {
            'api-key': self.api_key,
            'timestamp': timestamp,
            'signature': signature,
            'User-Agent': 'python-rest-client',
            'Content-Type': 'application/json'
        }
    
    def _check_rate_limit(self, weight: int = 1):
        """
        Check if request would exceed rate limits
        Resets window every 5 minutes
        """
        current_time = time.time()
        
        # Reset window if 5 minutes elapsed
        if current_time - self.window_start >= 300:
            self.request_count = 0
            self.window_start = current_time
        
        # Check quota
        if self.request_count + weight > self.config['api']['rate_limit_quota']:
            wait_time = 300 - (current_time - self.window_start)
            logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
            raise Exception(f"Rate limit exceeded. Wait {wait_time:.2f}s")
        
        self.request_count += weight
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                     data: Dict = None, weight: int = 1) -> Tuple[bool, Any]:
        """
        Make authenticated HTTP request to Delta Exchange
        
        Returns: (success: bool, response_data: dict or error_message: str)
        """
        # Check rate limit
        self._check_rate_limit(weight)
        
        # Build full URL
        url = f"{self.base_url}{endpoint}"
        
        # Prepare query string and payload
        query_string = ""
        if params:
            query_string = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        payload = ""
        if data:
            import json
            payload = json.dumps(data)
        
        # Generate headers
        headers = self._get_headers(method, endpoint, query_string, payload)
        
        try:
            logger.debug(f"{method} {url}{query_string}")
            
            response = requests.request(
                method,
                url,
                params=params,
                data=payload if payload else None,
                headers=headers,
                timeout=(self.config['timeouts']['connect'], 
                        self.config['timeouts']['read'])
            )
            
            # Handle HTTP errors
            if response.status_code == 429:
                reset_time = response.headers.get('X-RATE-LIMIT-RESET', '5000')
                logger.error(f"Rate limit hit. Reset in {reset_time}ms")
                return False, f"Rate limit exceeded. Wait {int(reset_time)/1000}s"
            
            response.raise_for_status()
            
            # Parse JSON response
            json_response = response.json()
            
            # Check API-level success
            if json_response.get('success'):
                logger.info(f"✓ {method} {endpoint} - Success")
                return True, json_response.get('result')
            else:
                error = json_response.get('error', {})
                error_code = error.get('code', 'unknown')
                error_context = error.get('context', {})
                logger.error(f"✗ {method} {endpoint} - API Error: {error_code}")
                return False, f"API Error: {error_code} - {error_context}"
        
        except requests.exceptions.Timeout:
            logger.error(f"✗ {method} {endpoint} - Timeout")
            return False, "Request timeout"
        
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ {method} {endpoint} - Request failed: {e}")
            return False, str(e)
        
        except Exception as e:
            logger.error(f"✗ {method} {endpoint} - Unexpected error: {e}")
            return False, str(e)
    
    # ============= PUBLIC ENDPOINTS =============
    
    def get_product_by_symbol(self, symbol: str) -> Tuple[bool, Any]:
        """
        Get product details by symbol (e.g., SOLUSD)
        Weight: 3
        """
        return self._make_request('GET', f'/v2/products/{symbol}', weight=3)
    
    def get_wallet_balances(self) -> Tuple[bool, Any]:
        """
        Get wallet balances
        Weight: 3
        """
        return self._make_request('GET', '/v2/wallet/balances', weight=3)
    
    # ============= ORDER MANAGEMENT =============
    
    def place_order(self, product_id: int, size: int, side: str, 
                   limit_price: str, order_type: str = "limit_order",
                   time_in_force: str = "gtc", 
                   post_only: bool = False,
                   reduce_only: bool = False,
                   client_order_id: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Place a new order
        Weight: 5
        
        Args:
            product_id: Product ID (e.g., for SOLUSD)
            size: Order size in contracts
            side: "buy" or "sell"
            limit_price: Price as string
            order_type: "limit_order" or "market_order"
            time_in_force: "gtc" or "ioc"
            post_only: True to ensure maker order
            reduce_only: True to only close positions
            client_order_id: Custom order ID (max 32 chars)
        """
        data = {
            "product_id": product_id,
            "size": size,
            "side": side,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "post_only": post_only,
            "reduce_only": reduce_only
        }
        
        if order_type == "limit_order":
            data["limit_price"] = limit_price
        
        if client_order_id:
            data["client_order_id"] = client_order_id[:32]  # Max 32 chars
        
        return self._make_request('POST', '/v2/orders', data=data, weight=5)
    
    def cancel_order(self, order_id: int, product_id: int) -> Tuple[bool, Any]:
        """
        Cancel an order by ID
        Weight: 5
        """
        data = {
            "id": order_id,
            "product_id": product_id
        }
        return self._make_request('DELETE', '/v2/orders', data=data, weight=5)
    
    def edit_order(self, order_id: int, product_id: int, 
                   limit_price: Optional[str] = None,
                   size: Optional[int] = None) -> Tuple[bool, Any]:
        """
        Edit an existing order
        Weight: 5
        """
        data = {
            "id": order_id,
            "product_id": product_id
        }
        
        if limit_price:
            data["limit_price"] = limit_price
        if size:
            data["size"] = size
        
        return self._make_request('PUT', '/v2/orders', data=data, weight=5)
    
    def cancel_all_orders(self, product_id: Optional[int] = None,
                         cancel_limit_orders: bool = True,
                         cancel_stop_orders: bool = True) -> Tuple[bool, Any]:
        """
        Cancel all open orders
        Weight: 5
        
        If product_id is None, cancels ALL orders across all products
        """
        data = {
            "cancel_limit_orders": cancel_limit_orders,
            "cancel_stop_orders": cancel_stop_orders
        }
        
        if product_id:
            data["product_id"] = product_id
        
        return self._make_request('DELETE', '/v2/orders/all', data=data, weight=5)
    
    def get_active_orders(self, product_id: Optional[int] = None,
                         states: str = "open,pending") -> Tuple[bool, Any]:
        """
        Get active orders
        Weight: 3
        
        Args:
            product_id: Filter by product (optional)
            states: Comma-separated states (default: "open,pending")
        """
        params = {"states": states}
        if product_id:
            params["product_id"] = product_id
        
        return self._make_request('GET', '/v2/orders', params=params, weight=3)
    
    def get_order_by_id(self, order_id: int) -> Tuple[bool, Any]:
        """
        Get order by ID
        Weight: 1
        """
        return self._make_request('GET', f'/v2/orders/{order_id}', weight=1)
    
    # ============= POSITION MANAGEMENT =============
    
    def get_position(self, product_id: int) -> Tuple[bool, Any]:
        """
        Get real-time position for a product
        Weight: 3
        
        Returns size and entry_price immediately
        """
        params = {"product_id": product_id}
        return self._make_request('GET', '/v2/positions', params=params, weight=3)
    
    def get_margined_positions(self, product_ids: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Get all margined positions
        Weight: 3
        
        May have up to 10s delay for updates
        Use get_position() for real-time data
        """
        params = {}
        if product_ids:
            params["product_ids"] = product_ids
        
        return self._make_request('GET', '/v2/positions/margined', params=params, weight=3)
    
    def close_all_positions(self, close_all_isolated: bool = True,
                           close_all_portfolio: bool = False) -> Tuple[bool, Any]:
        """
        Close all open positions
        Weight: 5
        """
        data = {
            "close_all_isolated": close_all_isolated,
            "close_all_portfolio": close_all_portfolio
        }
        return self._make_request('POST', '/v2/positions/close_all', data=data, weight=5)
    
    # ============= TRADE HISTORY =============
    
    def get_order_history(self, product_id: Optional[int] = None,
                         start_time: Optional[int] = None,
                         end_time: Optional[int] = None,
                         page_size: int = 100) -> Tuple[bool, Any]:
        """
        Get order history (closed and cancelled orders)
        Weight: 10
        
        VERY IMPORTANT for trade reconciliation
        
        Args:
            product_id: Filter by product
            start_time: From time in microseconds
            end_time: To time in microseconds  
            page_size: Records per page (default 100)
        """
        params = {"page_size": page_size}
        
        if product_id:
            params["product_id"] = product_id
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        return self._make_request('GET', '/v2/orders/history', params=params, weight=10)
    
    def get_fills(self, product_id: Optional[int] = None,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 page_size: int = 100) -> Tuple[bool, Any]:
        """
        Get user fills (executed trades)
        Weight: 10
        
        VERY IMPORTANT for trade reconciliation
        
        Returns actual executed trades with:
        - fill price
        - fill size
        - commission paid
        - role (maker/taker)
        """
        params = {"page_size": page_size}
        
        if product_id:
            params["product_id"] = product_id
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        return self._make_request('GET', '/v2/fills', params=params, weight=10)


# ============= TEST SUITE =============

def run_api_tests(api: DeltaExchangeAPI):
    """
    Comprehensive API test suite
    Tests all critical endpoints without risking real money
    """
    logger.info("="*60)
    logger.info("STARTING DELTA EXCHANGE API TEST SUITE")
    logger.info("="*60)
    
    results = {
        "passed": 0,
        "failed": 0,
        "tests": []
    }
    
    def log_test(name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
        if message:
            logger.info(f"  └─ {message}")
        
        results["tests"].append({
            "name": name,
            "success": success,
            "message": message
        })
        
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1
    
    # Test 1: Get Product Info
    logger.info("\n--- Test 1: Get Product Information ---")
    success, data = api.get_product_by_symbol("SOLUSD")
    log_test("Get SOLUSD Product", success, 
             f"Product ID: {data.get('id')}" if success else str(data))
    
    if not success:
        logger.error("Cannot proceed without product info")
        return results
    
    product_id = data['id']
    logger.info(f"SOLUSD Product ID: {product_id}")
    
    # Test 2: Get Wallet Balances
    logger.info("\n--- Test 2: Get Wallet Balances ---")
    success, data = api.get_wallet_balances()
    log_test("Get Wallet Balances", success,
             f"Found {len(data.get('result', []))} wallets" if success else str(data))
    
    # Test 3: Get Current Position
    logger.info("\n--- Test 3: Get Current Position ---")
    success, data = api.get_position(product_id)
    log_test("Get Position", success,
             f"Size: {data.get('size', 0)} @ {data.get('entry_price', 'N/A')}" if success else str(data))
    
    # Test 4: Get Active Orders (should be empty initially)
    logger.info("\n--- Test 4: Get Active Orders ---")
    success, data = api.get_active_orders(product_id)
    log_test("Get Active Orders", success,
             f"Found {len(data) if isinstance(data, list) else 0} orders" if success else str(data))
    
    # Test 5: Place Test Order
    logger.info("\n--- Test 5: Place Test Order (Post-Only) ---")
    # Use a price far from market to avoid fills
    success, data = api.place_order(
        product_id=product_id,
        size=1,
        side="buy",
        limit_price="50.0",  # Far below market
        order_type="limit_order",
        post_only=True,
        client_order_id="test_order_1"
    )
    log_test("Place Order", success,
             f"Order ID: {data.get('id')}" if success else str(data))
    
    if not success:
        logger.warning("Order placement failed - might be OK if insufficient margin")
        test_order_id = None
    else:
        test_order_id = data.get('id')
    
    # Test 6: Get Order By ID
    if test_order_id:
        logger.info("\n--- Test 6: Get Order By ID ---")
        success, data = api.get_order_by_id(test_order_id)
        log_test("Get Order By ID", success,
                 f"State: {data.get('state')}" if success else str(data))
        
        # Test 7: Edit Order
        logger.info("\n--- Test 7: Edit Order ---")
        success, data = api.edit_order(
            order_id=test_order_id,
            product_id=product_id,
            limit_price="51.0"
        )
        log_test("Edit Order", success,
                 f"New price: {data.get('limit_price')}" if success else str(data))
        
        # Test 8: Cancel Specific Order
        logger.info("\n--- Test 8: Cancel Order ---")
        success, data = api.cancel_order(test_order_id, product_id)
        log_test("Cancel Order", success,
                 f"Cancelled order {test_order_id}" if success else str(data))
    
    # Test 9: Cancel All Orders
    logger.info("\n--- Test 9: Cancel All Orders ---")
    success, data = api.cancel_all_orders(product_id)
    log_test("Cancel All Orders", success, "" if success else str(data))
    
    # Test 10: Get Order History
    logger.info("\n--- Test 10: Get Order History ---")
    # Get last 24 hours
    end_time = int(time.time() * 1000000)  # microseconds
    start_time = end_time - (24 * 60 * 60 * 1000000)  # 24h ago
    
    success, data = api.get_order_history(
        product_id=product_id,
        start_time=start_time,
        end_time=end_time,
        page_size=10
    )
    log_test("Get Order History", success,
             f"Found {len(data) if isinstance(data, list) else 0} historical orders" if success else str(data))
    
    # Test 11: Get Fills
    logger.info("\n--- Test 11: Get Fills ---")
    success, data = api.get_fills(
        product_id=product_id,
        start_time=start_time,
        end_time=end_time,
        page_size=10
    )
    log_test("Get Fills", success,
             f"Found {len(data) if isinstance(data, list) else 0} fills" if success else str(data))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUITE COMPLETE")
    logger.info(f"Passed: {results['passed']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Success Rate: {results['passed']/(results['passed']+results['failed'])*100:.1f}%")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    try:
        # Initialize API
        api = DeltaExchangeAPI("config.yml")
        
        # Run test suite
        results = run_api_tests(api)
        
        # Exit with appropriate code
        sys.exit(0 if results['failed'] == 0 else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
