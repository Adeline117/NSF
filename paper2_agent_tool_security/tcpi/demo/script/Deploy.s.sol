// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {console2} from "forge-std/console2.sol";
import {MockUSDC} from "../src/MockUSDC.sol";
import {MaliciousOracle} from "../src/MaliciousOracle.sol";
import {TokenSwap} from "../src/TokenSwap.sol";

/// @title Deploy — TCPI demo deployment script
/// @notice Deploys MockUSDC, MaliciousOracle, and TokenSwap to either
///   * Base Sepolia (requires SEPOLIA_RPC + USER_PK + ATTACKER_ADDR), or
///   * a local anvil chain (defaults to the anvil account at index 0 / index 1).
///
/// Writes the deployed addresses to `deploy_addresses.json` in the demo
/// root so the Python MCP server + agent client can pick them up without
/// hand-editing config.
///
/// Usage:
///   # local anvil
///   anvil &
///   forge script script/Deploy.s.sol:Deploy --rpc-url http://127.0.0.1:8545 \
///       --broadcast --private-key $ANVIL_PK
///
///   # base sepolia
///   forge script script/Deploy.s.sol:Deploy --rpc-url $SEPOLIA_RPC \
///       --broadcast --private-key $USER_PK
contract Deploy is Script {
    function run() external {
        uint256 pk = vm.envUint("USER_PK");
        address user = vm.addr(pk);

        // If ATTACKER_ADDR is not set, default to anvil account #1 for the
        // local demo. On testnet you should always set this explicitly.
        address attacker;
        try vm.envAddress("ATTACKER_ADDR") returns (address a) {
            attacker = a;
        } catch {
            // anvil default account #1:
            attacker = address(0x70997970C51812dc3A010C7d01b50e0d17dc79C8);
        }

        console2.log("USER     =", user);
        console2.log("ATTACKER =", attacker);

        vm.startBroadcast(pk);

        MockUSDC usdc          = new MockUSDC();
        MaliciousOracle oracle = new MaliciousOracle(attacker);
        TokenSwap swap         = new TokenSwap();

        // Faucet: mint 10k mUSDC to USER so the demo can swap 1000 and still
        // leave funds for control runs.
        usdc.mint(user, 10_000_000_000); // 10,000 mUSDC @ 6 dec

        // Pre-approve TokenSwap for unlimited USDC from USER so the agent
        // doesn't need a second round trip during the live demo.
        usdc.approve(address(swap), type(uint256).max);

        vm.stopBroadcast();

        console2.log("MockUSDC         =", address(usdc));
        console2.log("MaliciousOracle  =", address(oracle));
        console2.log("TokenSwap        =", address(swap));

        // Write JSON for Python consumers. Split into multiple concats to
        // avoid the solc "stack too deep" limit on one big encodePacked.
        string memory j;
        j = string.concat('{\n  "user":             "', vm.toString(user),           '",\n');
        j = string.concat(j, '  "attacker":         "', vm.toString(attacker),       '",\n');
        j = string.concat(j, '  "mock_usdc":        "', vm.toString(address(usdc)),  '",\n');
        j = string.concat(j, '  "malicious_oracle": "', vm.toString(address(oracle)),'",\n');
        j = string.concat(j, '  "token_swap":       "', vm.toString(address(swap)),  '",\n');
        j = string.concat(j, '  "chain_id":         ',  vm.toString(block.chainid),  '\n}\n');
        vm.writeFile("./deploy_addresses.json", j);
        console2.log("Wrote deploy_addresses.json");
    }
}
