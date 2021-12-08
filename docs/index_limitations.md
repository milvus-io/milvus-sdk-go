# Index Parameter Range

Searching with most indexes that Milvus supported requires specifying construction and search parameters. Listed below are the type and ranges of these parameters.

<table class="index_limit">
	<thead>
	<tr>
		<th>Index</th>
		<th>Type</th>
		<th>Const. Param & Range</th>
		<th>Search Param & Range</th>
    <th>Note</th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td>Flat</td>
		<td>entity.Flat</td>
		<td>N/A</td>
		<td>N/A</td>
 		<td>No parameter is required for search with Flat.</td>
	</tr>
	<tr>
		<td>BinFlat</td>
		<td>entity.BinFlat</td>
		<td><code>nlist</code>&isin;[1, 65536]</td>
		<td><code>nprobe</code>&isin;[1, <code>nlist</code>]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>IvfFlat</td>
		<td>entity.IvfFlat</td>
		<td><code>nlist</code>&isin;[1, 65536]</td>
		<td><code>nprobe</code>&isin;[1, <code>nlist</code>]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>BinIvfFlat</td>
		<td>entity.BinIvfFlat</td>
		<td><code>nlist</code>&isin;[1, 65536]</td>
		<td><code>nprobe</code>&isin;[1, <code>nlist</code>]</td>
		<td>BinIvfFlat will be supported in upcoming version of Milvus.</td>
	</tr>
	<tr>
		<td>IvfSQ8</td>
		<td>entity.IvfSQ8</td>
		<td><code>nlist</code>&isin;[1, 65536]</td>
		<td><code>nprobe</code>&isin;[1, <code>nlist</code>]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>IvfSQ8H</td>
		<td>entity.IvfSQ8H</td>
		<td><code>nlist</code>&isin;[1, 65536]</td>
		<td><code>nprobe</code>&isin;[1, <code>nlist</code>]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>IvfPQ</td>
		<td>entity.IvfPQ</td>
		<td><code>nlist</code>&isin;[1, 65536]<br/><code>m</code> dim===0 (mod self)<br/><code>nbits</code>&isin;[1, 16]</td>
		<td><code>nprobe</code>&isin;[1, <code>nlist</code>]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>RNSG</td>
		<td>entity.NSG</td>
		<td><code>out_degree</code>&isin;[5, 300]<br/><code>candidate_pool_size</code>&isin;[50, 1000]<br/><code>search_length</code>&isin;[10, 300]<br/><code>knng</code>&isin;[5, 300]</td>
		<td><code>search_length</code>&isin;[10, 300]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>HNSW</td>
		<td>entity.HNSW</td>
		<td><code>M</code>&isin;[4, 64]<br/><code>efConstruction</code>&isin;[8, 512]</td>
		<td><code>ef</code>&isin;[topK, 32768]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>RHNSWFlat</td>
		<td>entity.RHNSWFlat</td>
		<td><code>M</code>&isin;[4, 64]<br/><code>efConstruction</code>&isin;[8, 512]</td>
		<td><code>ef</code>&isin;[topK, 32768]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>RHNSW_PQ</td>
		<td>entity.RHNSW_PQ</td>
		<td><code>M</code>&isin;[4, 64]<br/><code>efConstruction</code>&isin;[8, 512]<br/><code>PQM</code> dim===0 (mod self)</td>
		<td><code>ef</code>&isin;[topK, 32768]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>RHNSW_SQ</td>
		<td>entity.RHNSWSQ</td>
		<td><code>M</code>&isin;[4, 64]<br/><code>efConstruction</code>&isin;[8, 512]</td>
		<td><code>ef</code>&isin;[topK, 32768]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>IvfHNSW</td>
		<td>entity.IvfHNSW</td>
		<td><code>nlist</code>&isin;[1, 65536]<br/><code>M</code>&isin;[4, 64]<br/><code>efConstruction</code>&isin;[8, 512]</td>
		<td><code>nprobe</code>&isin;[1, <code>nlist</code>]<br/><code>ef</code>&isin;[topK, 32768]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>ANNOY</td>
		<td>entity.ANNOY</td>
		<td><code>n_trees</code>&isin;[1, 1024]</td>
		<td><code>search_k</code>&isin;-1 or [topk, n * n_trees]</td>
		<td>&nbsp;</td>
	</tr>
	<tr>
		<td>NGTPANNG</td>
		<td>entity.NGTPANNG</td>
		<td><code>edge_size</code>&isin;[1, 200]<br/><code>forcedly_pruned_edge_size</code>&isin;[selectively_pruned_edge_size + 1, 200]<br/><code>selectively_pruned_edge_size</code>&isin;[1, forcedly_pruned_edge_size -1 ]</td>
		<td><code>max_search_edges</code>&isin;[-1, 200]<br/><code>epsilon</code>&isin;[-1.0, 1.0]</td>
		<td>Search parameter epsilon type is float64.</td>
	</tr>
	<tr>
		<td>NGTONNG</td>
		<td>entity.NGTONNG</td>
		<td><code>edge_size</code>&isin;[1, 200]<br/><code>outgoing_edge_size</code>&isin;[1, 200]<br/><code>incoming_edge_size</code>&isin;[1, 200]</td>
		<td><code>max_search_edges</code>&isin;[-1, 200]<br/><code>epsilon</code>&isin;[-1.0, 1.0]</td>
		<td>Search parameter epsilon type is float64.</td>
	</tr>
	</tbody>
</table>

