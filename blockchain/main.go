package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
)

type block struct {
	previousHash []byte
	hash         []byte
	data         []byte
}

type BlockChain struct {
	blocks []*block
}

func (b *block) calcHash() {
	info := bytes.Join([][]byte{b.data, b.previousHash}, []byte{})
	hash := sha256.Sum256(info)
	b.hash = hash[:]
}

func makeBlock(data string, previousHash []byte) *block {
	block := &block{[]byte{}, []byte(data), previousHash}
	block.calcHash()
	return block
}

func (chain *BlockChain) addBlock(data string) {
	previousBlock := chain.blocks[len(chain.blocks)-1]
	new := makeBlock(data, previousBlock.hash)
	chain.blocks = append(chain.blocks, new)
}

func Genesis() *block {
	return makeBlock("Genesis", []byte{})
}

func InitBlockChain() *BlockChain {
	return &BlockChain{[]*block{Genesis()}}
}

func main() {
	chain := InitBlockChain()

	for i := 0; i < len(chain.blocks); i++ {
		previousHash := chain.blocks[i].previousHash
		data := chain.blocks[i].data
		hash := chain.blocks[i].hash

		fmt.Printf("Previous Hash: %x\n", previousHash)
		fmt.Printf("Data in Block: %s\n", data)
		fmt.Printf("Hash: %x\n", hash)
	}
}
