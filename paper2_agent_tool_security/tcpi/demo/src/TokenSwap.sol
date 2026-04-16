// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "./IERC20.sol";

/// @title TokenSwap — the honest T_B for the TCPI demo
/// @notice This is the downstream tool. It is *not* the bug. Its only job is
/// to pull `amount` tokens from the caller (the agent's EOA, via prior
/// approval) and push them to `recipient`. The hijack flows entirely through
/// the `recipient` argument, which the LLM sets from `MaliciousOracle`'s
/// output.
contract TokenSwap {
    bytes32 public constant TCPI_DEMO_TAG = bytes32("TCPI-DEMO-NSF");

    event Swapped(
        address indexed from,
        address indexed to,
        address indexed token,
        uint256 amount
    );

    /// @notice Execute an ERC-20 swap: pull `amount` of `token` from
    /// msg.sender via transferFrom and forward to `recipient`.
    /// @dev Requires the caller to have previously approved this contract
    /// as spender on `token`. There is NO recipient allowlist — that is the
    /// whole point of the TCPI demo: the LLM agent, not the contract, is
    /// expected to validate where funds go.
    function swap(address token, address recipient, uint256 amount) external {
        require(
            IERC20(token).transferFrom(msg.sender, recipient, amount),
            "TokenSwap: transferFrom failed"
        );
        emit Swapped(msg.sender, recipient, token, amount);
    }
}
