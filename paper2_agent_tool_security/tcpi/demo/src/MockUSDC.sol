// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "./IERC20.sol";

/// @notice Minimal ERC-20 token used only for the TCPI demo on local anvil
/// or Base Sepolia. 6 decimals, mimicking USDC. NOT for production use.
contract MockUSDC is IERC20 {
    string public constant name = "Mock USDC (TCPI demo)";
    string public constant symbol = "mUSDC";
    uint8 public constant decimals = 6;
    // Watermark so any mis-deployed copy is attributable to this demo.
    bytes32 public constant TCPI_DEMO_TAG = bytes32("TCPI-DEMO-NSF");

    uint256 private _totalSupply;
    mapping(address => uint256) private _balanceOf;
    mapping(address => mapping(address => uint256)) private _allowance;

    function totalSupply() external view returns (uint256) { return _totalSupply; }
    function balanceOf(address a) external view returns (uint256) { return _balanceOf[a]; }
    function allowance(address o, address s) external view returns (uint256) { return _allowance[o][s]; }

    function mint(address to, uint256 amount) external {
        // Unrestricted mint so test scripts/faucets can top up any wallet.
        _balanceOf[to] += amount;
        _totalSupply += amount;
        emit Transfer(address(0), to, amount);
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        _allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 a = _allowance[from][msg.sender];
        require(a >= amount, "ERC20: insufficient allowance");
        if (a != type(uint256).max) {
            _allowance[from][msg.sender] = a - amount;
        }
        _transfer(from, to, amount);
        return true;
    }

    function _transfer(address from, address to, uint256 amount) internal {
        require(_balanceOf[from] >= amount, "ERC20: insufficient balance");
        unchecked { _balanceOf[from] -= amount; }
        _balanceOf[to] += amount;
        emit Transfer(from, to, amount);
    }
}
